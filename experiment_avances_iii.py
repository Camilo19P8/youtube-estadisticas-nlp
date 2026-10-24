import os, sys
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'python-deps'))
import json, hashlib, html, time, platform, warnings
import numpy as np, pandas as pd, sklearn, scipy, joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from threadpoolctl import threadpool_limits
warnings.filterwarnings('ignore',category=UserWarning)
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
BASE=['log_likes','log_comments','age_days','title_chars','title_words']
RED=[c for c in BASE if c!='log_comments']
AGG=['n_sample','mean_comment_likes','s0','s1','s2','mean_chars','exclaim_rate']

def load_data():
    v=pd.read_csv(ROOT/'data/videos-stats.csv'); c=pd.read_csv(ROOT/'data/comments.csv')
    audit={'videos_raw':len(v),'comments_raw':len(c),'video_duplicates':int(v['Video ID'].duplicated().sum()),'missing_text':int(c.Comment.isna().sum())}
    v=v.drop_duplicates('Video ID').dropna(subset=['Views']).copy()
    for col in ['Likes','Comments']:
        v.loc[v[col]<0,col]=np.nan
    c=c.dropna(subset=['Comment']).copy(); before=len(c)
    c=c.drop_duplicates(['Video ID','Comment']); audit['comment_duplicates']=before-len(c)
    c['Comment']=c.Comment.map(html.unescape)
    c['chars']=c.Comment.str.len(); c['exclaim']=c.Comment.str.contains('!',regex=False).astype(float)
    a=c.groupby('Video ID').agg(comment_text=('Comment',lambda s:' '.join(s)),n_sample=('Comment','size'),mean_comment_likes=('Likes','mean'),mean_chars=('chars','mean'),exclaim_rate=('exclaim','mean'))
    for s in range(3):
        a['s'+str(s)]=c.assign(ind=(c.Sentiment==s).astype(float)).groupby('Video ID')['ind'].mean()
    v=v.join(a,on='Video ID').reset_index(drop=True)
    v['Title']=v.Title.fillna('').map(html.unescape)
    v['comment_text']=v.comment_text.fillna('')
    v['combined_text']=v.Title+' '+v.comment_text
    v['age_days']=(pd.Timestamp('2022-08-24')-pd.to_datetime(v['Published At'])).dt.days
    v['title_chars']=v.Title.str.len();v['title_words']=v.Title.str.split().str.len()
    v['log_likes']=np.log1p(v.Likes);v['log_comments']=np.log1p(v.Comments)
    audit.update(videos_clean=len(v),comments_clean=len(c),keywords=v.Keyword.nunique(),missing_likes=int(v.Likes.isna().sum()),missing_comments=int(v.Comments.isna().sum()))
    return v,np.log1p(v.Views.to_numpy()),audit

def prep(nums=BASE,text=None):
    steps=[]
    if nums:
        steps.append(('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),nums))
        steps.append(('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['Keyword']))
    if text:
        steps.append(('text',TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=3000,sublinear_tf=True),text))
    return ColumnTransformer(steps,sparse_threshold=0.3 if text else 0)

def hgb(nums):
    return Pipeline([('prep',prep(nums)),('model',HistGradientBoostingRegressor(max_iter=150,learning_rate=.1,max_leaf_nodes=15,random_state=42))])

def candidates():
    r=lambda nums,text: Pipeline([('prep',prep(nums,text)),('model',Ridge(solver='lsqr'))])
    red=hgb(RED); hybrid=r(BASE,'combined_text')
    return [
      ('Mediana',Pipeline([('prep',prep()),('model',DummyRegressor(strategy='median'))]),{}),
      ('Ridge tabular',r(BASE,None),{'model__alpha':[1,10]}),
      ('Ridge comentarios',r(BASE,'comment_text'),{'model__alpha':[10,100]}),
      ('Ridge titulo y comentarios',hybrid,{'model__alpha':[10,100]}),
      ('Ridge solo comentarios',r([], 'comment_text'),{'model__alpha':[10,100]}),
      ('HGB completo',hgb(BASE),{'model__l2_regularization':[0,1]}),
      ('HGB sin Comments',red,{'model__l2_regularization':[0,1]}),
      ('HGB agregados NLP',hgb(BASE+AGG),{'model__l2_regularization':[0,1]}),
      ('Ensamble HGB y Ridge',VotingRegressor([('hgb',red),('ridge',hybrid)]),{})]

def metrics(y,p):
    raw=np.expm1(y); pred=np.maximum(0,np.expm1(p))
    return dict(rmse=float(root_mean_squared_error(y,p)),mae=float(mean_absolute_error(y,p)),r2=float(r2_score(y,p)),mae_views=float(mean_absolute_error(raw,pred)),rmse_views=float(root_mean_squared_error(raw,pred)))

def export_pipe(p):
    if isinstance(p,VotingRegressor):return dict(kind='voting',models=[export_pipe(x) for x in p.estimators_])
    pre=p.named_steps['prep']; m=p.named_steps['model']; enc={}
    for name,t,cols in pre.transformers_:
        if name=='num':
            imp=t.named_steps['impute']; sc=t.named_steps['scale']
            enc['num']=dict(cols=cols,medians=imp.statistics_.tolist(),indicators=imp.indicator_.features_.tolist(),mean=sc.mean_.tolist(),scale=sc.scale_.tolist())
        elif name=='cat':enc['categories']=t.categories_[0].tolist()
        elif name=='text':enc['text']=dict(col=cols,vocab={k:int(v) for k,v in t.vocabulary_.items()},idf=t.idf_.tolist())
    if isinstance(m,Ridge):return dict(kind='ridge',prep=enc,coef=m.coef_.tolist(),intercept=float(m.intercept_))
    if isinstance(m,DummyRegressor):return dict(kind='constant',value=float(m.constant_[0,0]))
    trees=[]
    for stage in m._predictors:
        nodes=stage[0].nodes
        trees.append([{k:(float(n[k]) if k in ['value','num_threshold'] else int(n[k])) for k in ['value','feature_idx','num_threshold','missing_go_to_left','left','right','is_leaf']} for n in nodes])
    return dict(kind='hgb',prep=enc,baseline=float(m._baseline_prediction[0,0]),trees=trees)

def run():
    t=time.perf_counter(); X,y,audit=load_data()
    train,oldtest=train_test_split(np.arange(len(X)),test_size=.2,random_state=42)
    # La prueba de Avances II ya se conoce y queda excluida de esta iteracion.
    X=X.iloc[train].reset_index(drop=True);y=y[train]
    X[['Video ID']].assign(partition='development').to_csv(OUT/'development_ids.csv',index=False)
    defs=candidates(); outer=KFold(5,shuffle=True,random_state=2026)
    foldrows=[]; preds={name:np.zeros(len(y)) for name,_,_ in defs}; procedure=np.zeros(len(y)); selections=[]
    for fold,(tr,va) in enumerate(outer.split(X),1):
        searches=[]
        for name,model,grid in defs:
            search=GridSearchCV(model,grid,scoring='neg_root_mean_squared_error',cv=KFold(3,shuffle=True,random_state=100+fold),n_jobs=1,return_train_score=True)
            start=time.perf_counter();search.fit(X.iloc[tr],y[tr]);p=search.predict(X.iloc[va]);preds[name][va]=p
            row=dict(fold=fold,model=name,inner_rmse=-search.best_score_,seconds=time.perf_counter()-start,params=json.dumps(search.best_params_),**metrics(y[va],p))
            foldrows.append(row);searches.append((search.best_score_,name,search,p))
            pd.DataFrame(foldrows).to_csv(OUT/'fold_metrics.csv',index=False)
            print(f'Fold {fold} | {name} | RMSE {row["rmse"]:.4f}',flush=True)
        best=max(searches,key=lambda z:z[0]);procedure[va]=best[3];selections.append(dict(fold=fold,model=best[1],inner_rmse=-best[0]))
    rows=[]; folds=pd.DataFrame(foldrows)
    for name,_,_ in defs:
        f=folds[folds.model==name]
        rows.append(dict(model=name,cv_mean=f.rmse.mean(),cv_sd=f.rmse.std(ddof=1),seconds=f.seconds.mean(),**metrics(y,preds[name])))
    summary=pd.DataFrame(rows).sort_values('cv_mean');summary.to_csv(OUT/'comparison.csv',index=False)
    # Seleccion final por CV interna en desarrollo completo, nunca por prueba antigua.
    finalsearch=[]
    for name,model,grid in defs:
        s=GridSearchCV(model,grid,scoring='neg_root_mean_squared_error',cv=KFold(3,shuffle=True,random_state=777),n_jobs=1)
        s.fit(X,y);finalsearch.append((s.best_score_,name,s))
    best=max(finalsearch,key=lambda z:z[0]); champion=best[2].best_estimator_
    joblib.dump(champion,OUT/'model.joblib')
    textmodel=next(s.best_estimator_ for _,n,s in finalsearch if n=='Ridge solo comentarios')
    joblib.dump(textmodel,OUT/'text_model.joblib')
    portable=dict(selected=best[1],champion=export_pipe(champion),text_model=export_pipe(textmodel),keywords=sorted(X.Keyword.unique().tolist()))
    (OUT/'portable_model.json').write_text(json.dumps(portable,ensure_ascii=False),encoding='utf-8')
    rng=np.random.default_rng(2026);boot=[]
    for _ in range(2000):
        idx=rng.integers(0,len(y),len(y));boot.append(root_mean_squared_error(y[idx],procedure[idx]))
    # Diferencia apareada entre cada candidato y HGB completo, con modelos OOF fijos.
    diffs={}
    for name in preds:
        vals=[]
        for _ in range(1000):
            ix=rng.integers(0,len(y),len(y));vals.append(root_mean_squared_error(y[ix],preds[name][ix])-root_mean_squared_error(y[ix],preds['HGB completo'][ix]))
        diffs[name]=np.quantile(vals,[.025,.975]).tolist()
    # Diagnostico por cuartil y tema del procedimiento de seleccion anidado.
    diag=pd.DataFrame({'y':y,'pred':procedure,'Keyword':X.Keyword})
    diag['quartile']=pd.qcut(diag.y,4,labels=['Q1','Q2','Q3','Q4'])
    segments=[]
    for q,g in diag.groupby('quartile',observed=True):segments.append(dict(segment=str(q),n=len(g),**metrics(g.y,g.pred)))
    pd.DataFrame(segments).to_csv(OUT/'segments.csv',index=False)
    pd.DataFrame({'Video ID':X['Video ID'],'y_log':y,'procedure_pred_log':procedure,**{n:p for n,p in preds.items()}}).to_csv(OUT/'oof_predictions.csv',index=False)
    sample=X.iloc[0]; request={k:(None if pd.isna(sample[k]) else sample[k].item() if hasattr(sample[k],'item') else sample[k]) for k in BASE+AGG+['Keyword','Title','comment_text','combined_text']}
    expected=float(champion.predict(X.iloc[[0]])[0]); text_expected=float(textmodel.predict(X.iloc[[0]])[0])
    (OUT/'parity_case.json').write_text(json.dumps(dict(input=request,expected=expected,text_expected=text_expected),ensure_ascii=False),encoding='utf-8')
    cases=[]
    for i in range(50):
        sample=X.iloc[i].copy()
        if i==48:sample['log_likes']=np.nan;sample['log_comments']=np.nan
        if i==49:sample['Keyword']='unseen-keyword'
        req={k:(None if pd.isna(sample[k]) else sample[k].item() if hasattr(sample[k],'item') else sample[k]) for k in BASE+AGG+['Keyword','Title','comment_text','combined_text']}
        frame=sample.to_frame().T
        for col in BASE+AGG:frame[col]=pd.to_numeric(frame[col])
        cases.append(dict(input=req,expected=float(champion.predict(frame)[0]),text_expected=float(textmodel.predict(frame)[0])))
    (OUT/'parity_cases.json').write_text(json.dumps(cases,ensure_ascii=False),encoding='utf-8')
    info=dict(audit=audit,n_development=len(y),n_oldtest=len(oldtest),selected=best[1],selected_params=best[2].best_params_,final_inner_rmse=-best[0],nested_selection=metrics(y,procedure),nested_bootstrap=np.quantile(boot,[.025,.975]).tolist(),outer_selections=selections,paired_ci=diffs,versions=dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'data').glob('*.csv')},seconds=time.perf_counter()-t,configs=sum(len(list(__import__('sklearn').model_selection.ParameterGrid(g))) for _,_,g in defs))
    (OUT/'summary.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,ax=plt.subplots(figsize=(8,4.6));z=summary.sort_values('cv_mean',ascending=False);ax.barh(z.model,z.cv_mean,color='#406879');ax.errorbar(z.cv_mean,z.model,xerr=z.cv_sd,fmt='none',ecolor='black',capsize=3);ax.set_xlabel('RMSE log medio en 5 particiones externas');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3.7));axs[0].scatter(y,procedure,s=8,alpha=.45,color='#406879');axs[0].plot([y.min(),y.max()],[y.min(),y.max()],'k--',lw=1);axs[0].set(xlabel='Observado ln(1 + vistas)',ylabel='Prediccion externa');axs[1].scatter(procedure,y-procedure,s=8,alpha=.45,color='#406879');axs[1].axhline(0,color='black',ls='--',lw=1);axs[1].set(xlabel='Prediccion externa',ylabel='Residuo log');fig.tight_layout();fig.savefig(OUT/'diagnostics.png',dpi=180);plt.close(fig)
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
