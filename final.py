
# coding: utf-8

# In[1]:

## Import warnings. Supress warnings (for  matplotlib)
import warnings
warnings.simplefilter("ignore")


# In[73]:



## Import analysis modules
import pandas as p
from pandas.tools.plotting import scatter_matrix
from numpy import nan, isnan, mean, std, hstack, ravel
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut, LeavePOut, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Binarizer, Imputer, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import learning_curve, GridSearchCV
## Import visualization modules
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

## Import SciPy
from scipy.sparse import issparse


# In[3]:

## Read in file
data = p.read_csv('/Users/ChunyangLi/loan.csv',delimiter='~}',na_values='nan',)


# In[4]:

## Count of instances and features
rows, columns = data.shape
print data.shape


# In[5]:

## Get basic statistics for continuous features
numeric = data.describe(include=['number']).T.reset_index()
numeric.rename(columns={'index':'feature'},inplace=True)
numeric.insert(1,'missing',(rows - numeric['count'])/ float(rows))


# In[6]:

## How many features can we eliminate?
drop = numeric[(numeric['missing']==1) | (numeric['std']==0)]


# In[7]:

## Drop the unhelpful features from the base and numeric table
data = data.drop(drop['feature'],axis=1)
numeric = numeric[~numeric['feature'].isin(drop['feature'])]


# In[8]:

## Get basic statistics for discrete features
discrete = data.describe(include=['object']).T.reset_index()
discrete.rename(columns={'index':'feature'},inplace=True)
discrete.insert(1,'missing',(rows - discrete['count'])/ float(rows))


# In[9]:

## How many features can we eliminate?
ddrop = discrete[(discrete['missing']>.6) | (discrete['unique']==1)]


# In[10]:

## Drop unhelpful features from the base table
data = data.drop(ddrop['feature'],axis=1)
discrete = discrete[~discrete['feature'].isin(ddrop['feature'])]


# In[11]:

## How many columns do we have left?
data.shape


# In[12]:

## Double check discrete
discrete


# In[13]:

## Discrete remove
data = data.drop(['grade','sub_grade','int_rate','emp_title','issue_d','pymnt_plan','url','desc','title','earliest_cr_line','last_pymnt_d','last_credit_pull_d'],axis=1)


# In[14]:

## Check numeric
numeric


# In[15]:

data = data.drop(['id','member_id'],axis=1)


# In[16]:

data.shape


# In[17]:

## Keep only those loan statuses where fully paid or charged off
data = data[data['loan_status'].isin(['Fully Paid','Charged Off'])]


# In[18]:

sparse_cols = ['delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec',
          'total_acc','out_prncp','out_prncp_inv','total_rec_late_fee','recoveries','collection_recovery_fee', 
          'acc_now_delinq','delinq_amnt','pub_rec_bankruptcies', 'tax_liens']


# In[19]:

discrete_cols = ['term','emp_length','home_ownership','verification_status', 'purpose', 
            'zip_code', 'addr_state']


# In[20]:

numeric_cols = [x for x in data.columns if x not in sparse_cols + discrete_cols + ['loan_status']]


# In[21]:

## Address by stripping leading space
data['term'] = data['term'].str.strip()


# In[22]:

## Scikit learn estimators require numeric features
term_map = {'36 months':0,'60 months':1}
emp_map = {'n/a':0,'< 1 year':1,'1 year':2,'2 years':3,'3 years':4,'4 years':5,'5 years':6,'6 years':7,'7 years':8,'8 years':9,
           '9 years':10, '10 years':11}
status_map = {'Fully Paid':0,'Charged Off':1}


# In[23]:

## Convert categorical features to numeric using mapping function
data['term'] = data['term'].map(term_map)
data['emp_length'] = data['emp_length'].map(emp_map)
data['loan_status'] = data['loan_status'].map(status_map)


# In[24]:

data['emp_length'].fillna(0.0, inplace=True)


# In[25]:

## Leverage regular expressions to clean revol_util and int_rate
data['revol_util'].replace('%','',regex=True,inplace=True)


# In[26]:

## Convert revol_util to numeric 
data['revol_util'] = p.to_numeric(data['revol_util'])


# In[27]:

data['revol_util'].fillna(0.0, inplace=True)


# In[28]:

validation = data.sample(frac=.2,random_state=12345)
val_x = validation.drop('loan_status',axis=1)
val_y = validation['loan_status'].as_matrix()


# In[29]:

new_data = data.drop(validation.index,axis=0)


# In[30]:

## Seperate input features from target feature
x = new_data.drop('loan_status',axis=1)
y = new_data['loan_status'].as_matrix()


# In[31]:

## Take a look at x
x.head()


# In[32]:

## Take a look at y
y


# In[33]:

## Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.3,random_state=15254)


# In[34]:

## Impute missing cases using scikit learn
imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
for col in sparse_cols:
    imp.fit(x_train[col].values.reshape(-1,1))
    x_train[col] = imp.transform(x_train[col].values.reshape(-1,1))
    x_test[col] = imp.transform(x_test[col].values.reshape(-1,1))
    val_x[col] = imp.transform(val_x[col].values.reshape(-1,1))


# In[35]:

mas = MaxAbsScaler()
for col in sparse_cols:
    mas.fit(x_train[col].values.reshape(-1,1))
    x_train[col] = mas.transform(x_train[col].values.reshape(-1,1))
    x_test[col] = mas.transform(x_test[col].values.reshape(-1,1))
    val_x[col] = mas.transform(val_x[col].values.reshape(-1,1))


# In[36]:

cols = ['home_ownership','verification_status','purpose','zip_code','addr_state']
le = LabelEncoder()
for col in cols:
    le.fit(ravel(data[col]))
    data[col] = le.transform(ravel(data[col]))
    x_train[col] = le.transform(ravel(x_train[col]))
    x_test[col] = le.transform(ravel(x_test[col]))
    val_x[col] = le.transform(ravel(val_x[col]))


# In[37]:

## Standard histograms with pandas
rb=RobustScaler()
st=StandardScaler()
for col in numeric_cols:
    if col in ['annual_inc','revol_bal']:
        rb.fit(x_train[col].values.reshape(-1,1))
        x_train[col] = rb.transform(x_train[col].values.reshape(-1,1))
        x_test[col] = rb.transform(x_test[col].values.reshape(-1,1))
        val_x[col] = rb.transform(val_x[col].values.reshape(-1,1))
    else:
        st.fit(x_train[col].values.reshape(-1,1))
        x_train[col] = st.transform(x_train[col].values.reshape(-1,1))
        x_test[col] = st.transform(x_test[col].values.reshape(-1,1))
        val_x[col] = st.transform(val_x[col].values.reshape(-1,1))


# In[38]:

x_train.head()


# In[39]:

ohe = OneHotEncoder()
ohe.fit(data.loc[:,discrete_cols])
x_train_discrete = ohe.transform(x_train.loc[:,discrete_cols]).toarray()
x_test_discrete = ohe.transform(x_test.loc[:,discrete_cols]).toarray()
val_discrete = ohe.transform(val_x.loc[:,discrete_cols]).toarray()


# In[40]:

## Lets's try to extract components via PCA 
pca = PCA(n_components=5)
pca.fit(x_train.loc[:,numeric_cols])


# In[41]:

## Percentage of variance explained by each of the selected components.
print(['%0.2f' % z for z in pca.explained_variance_ratio_]) 


# In[42]:

## Transform x
x_train_pca = pca.transform(x_train.loc[:,numeric_cols])
x_test_pca = pca.transform(x_test.loc[:,numeric_cols])
val_x_pca = pca.transform(val_x.loc[:,numeric_cols])


# In[43]:

x_train_sparse = x_train.loc[:,sparse_cols].as_matrix()
x_test_sparse = x_test.loc[:,sparse_cols].as_matrix()
val_x_sparse = val_x.loc[:,sparse_cols].as_matrix()


# In[44]:

x_train_final = hstack([x_train_pca,x_train_discrete,x_train_sparse])
x_test_final = hstack([x_test_pca,x_test_discrete,x_test_sparse])
val_x_final = hstack([val_x_pca,val_discrete,val_x_sparse])


# In[45]:

print x_train_final.shape
print x_test_final.shape
print val_x_final.shape


# In[46]:

## Create estimator
clf = DecisionTreeClassifier(class_weight='balanced',max_depth=3)


# In[64]:

## Fit the model using training set 
clf.fit(x_train_final,y_train)


# In[65]:

## Check accuracy score
print '%0.2f' % clf.score(x_test_final,y_test)


# In[66]:

## Predict y given test set
predictions = clf.predict(x_test_final)


# In[67]:

## Take a look at the confusion matrix ([TN,FN],[FP,TP])
confusion_matrix(y_test,predictions)


# In[68]:

## Accuracy score
print '%0.2f' % precision_score(y_test, predictions)


# In[69]:

## Recall score
print '%0.2f' % recall_score(y_test, predictions)


# In[70]:

## Print classification report
print classification_report(y_test, predictions)


# In[71]:

## Get data to plot ROC Curve
fp, tp, th = roc_curve(y_test, predictions)
roc_auc = auc(fp, tp)


# In[72]:

## Plot ROC Curve
plt.title('ROC Curve')
plt.plot(fp, tp, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[56]:

## Check accuracy score
print '%0.2f' % clf.score(val_x_final,val_y)


# In[57]:

## Predict y given test set
predictions = clf.predict(val_x_final)


# In[58]:

## Take a look at the confusion matrix ([TN,FN],[FP,TP])
confusion_matrix(val_y,predictions)


# In[59]:

## Accuracy score
print '%0.2f' % precision_score(val_y, predictions)


# In[60]:

## Recall score
print '%0.2f' % recall_score(val_y, predictions)


# In[61]:

## Print classification report
print classification_report(val_y, predictions)


# In[62]:

## Get data to plot ROC Curve
fp, tp, th = roc_curve(val_y, predictions)
roc_auc = auc(fp, tp)


# In[63]:

## Plot ROC Curve
plt.title('ROC Curve')
plt.plot(fp, tp, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[75]:

params = {'class_weight':['balanced',None],
          'criterion':['gini','entropy'],
          'max_depth':[2,3,4],
          'max_features':['auto','sqrt','log2',None],
          'min_samples_leaf':[1,2,3],
          'min_samples_split':[2,4]
}


# In[81]:

## Create estimator
clf = DecisionTreeClassifier()


# In[99]:

gs = GridSearchCV(clf,param_grid=params,scoring='recall')


# In[100]:

gs.fit(x_train_final,y_train)


# In[87]:

gs.best_score_


# In[88]:

gs.best_estimator_


# In[89]:

gs.scorer_


# In[90]:

est = gs.best_estimator_


# In[91]:

## Fit the model using training set 
est.fit(x_train_final,y_train)


# In[93]:

## Check accuracy score
print '%0.2f' % est.score(x_test_final,y_test)


# In[94]:

## Predict y given test set
predictions = est.predict(x_test_final)


# In[95]:

## Take a look at the confusion matrix ([TN,FN],[FP,TP])
confusion_matrix(y_test,predictions)


# In[96]:

## Accuracy score
print '%0.2f' % precision_score(y_test, predictions)


# In[97]:

## Recall score
print '%0.2f' % recall_score(y_test, predictions)


# In[98]:

## Print classification report
print classification_report(y_test, predictions)


# In[71]:

## Get data to plot ROC Curve
fp, tp, th = roc_curve(y_test, predictions)
roc_auc = auc(fp, tp)


# In[72]:

## Plot ROC Curve
plt.title('ROC Curve')
plt.plot(fp, tp, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

