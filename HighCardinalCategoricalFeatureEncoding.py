# Three encoding method were used as features

# label encoding because i used tree base algorithms
LBL = preprocessing.LabelEncoder()

LE_vars=[]
LE_map=dict()
for cat_var in cat_vars:
    print ("Label Encoding %s" % (cat_var))
    LE_var=cat_var+'_le'
    full_data[LE_var]=LBL.fit_transform(full_data[cat_var])
    LE_vars.append(LE_var)
    LE_map[cat_var]=LBL.classes_
    
print ("Label-encoded feaures: %s" % (LE_vars))


# frequency encoding with prior and postirier weights
# this comes from discussion form
def designate_single_observations(df1, df2, column):
    ps = df1[column].append(df2[column])
    grouped = ps.groupby(ps).size().to_frame().rename(columns={0: "size"})
    df1.loc[df1.join(grouped, on=column, how="left")["size"] <= 1, column] = -1
    df2.loc[df2.join(grouped, on=column, how="left")["size"] <= 1, column] = -1
    return df1, df2


def hcc_encode(train_df, test_df, variable, target, prior_prob, k, f=1, g=1, r_k=None, update_df=None):
    """
    See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems" by Daniele Micci-Barreca
    """
    hcc_name = "_".join(["hcc", variable, target])

    grouped = train_df.groupby(variable)[target].agg({"size": "size", "mean": "mean"})
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
    grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

    df = test_df[[variable]].join(grouped, on=variable, how="left")[hcc_name].fillna(prior_prob)
    if r_k: df *= np.random.uniform(1 - r_k, 1 + r_k, len(test_df))     # Add uniform noise. Not mentioned in original paper

    if update_df is None: update_df = test_df
    if hcc_name not in update_df.columns: update_df[hcc_name] = np.nan
    update_df.update(df)
    return

 for col in ('building_id', 'manager_id', 'display_address'):
    train_data, test_data = designate_single_observations(train_data, test_data, col)
    
prior_low, prior_medium, prior_high = train_data[["low", "medium", "high"]].mean() 

skf = model_selection.StratifiedKFold(5)
attributes = product(("building_id", "manager_id"), zip(("medium", "high"), (prior_medium, prior_high)))
for variable, (target, prior) in attributes:
    hcc_encode(train_data, test_data, variable, target, prior, k=5, r_k=None)
    for train, test in skf.split(np.zeros(len(train_data)), train_data['interest_level']):
        hcc_encode(train_data.iloc[train], train_data.iloc[test], variable, target, prior, k=5, r_k=0.01,
                   update_df=train_data)
        
hcc_data = pd.concat([train_data[['building_id', 'manager_id', 'display_address',
            'hcc_building_id_medium','hcc_building_id_high',
            'hcc_manager_id_medium','hcc_manager_id_high']],
           test_data[['building_id', 'manager_id', 'display_address',
            'hcc_building_id_medium','hcc_building_id_high',
            'hcc_manager_id_medium','hcc_manager_id_high']]
           ]
          )
full_data['building_id'] = hcc_data['building_id']
full_data['manager_id'] = hcc_data['manager_id']
full_data['display_address'] = hcc_data['display_address']
full_data['hcc_building_id_medium'] = hcc_data['hcc_building_id_medium']
full_data['hcc_building_id_high'] = hcc_data['hcc_building_id_high']
full_data['hcc_manager_id_medium'] = hcc_data['hcc_manager_id_medium']
full_data['hcc_manager_id_high'] = hcc_data['hcc_manager_id_high']
hcc_vars = ['hcc_building_id_medium','hcc_building_id_high','hcc_manager_id_medium','hcc_manager_id_high']    


# another frequency based encoding method from discussion board - focus on the interaction of manager and buildings
import random
index=list(range(train_data.shape[0]))
random.shuffle(index)
a=[np.nan]*len(train_data)
b=[np.nan]*len(train_data)
c=[np.nan]*len(train_data)

for i in range(5):
    building_level={}
    for j in train_data['manager_id'].values:
        building_level[j]=[0,0,0]
    test_index=index[int((i*train_data.shape[0])/5):int(((i+1)*train_data.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    for j in train_index:
        temp=train_data.iloc[j]
        if temp['interest_level']=='low':
            building_level[temp['manager_id']][0]+=1
        if temp['interest_level']=='medium':
            building_level[temp['manager_id']][1]+=1
        if temp['interest_level']=='high':
            building_level[temp['manager_id']][2]+=1
    for j in test_index:
        temp=train_data.iloc[j]
        if sum(building_level[temp['manager_id']])!=0:
            a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])
            b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])
            c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])
train_data['manager_level_low']=a
train_data['manager_level_medium']=b
train_data['manager_level_high']=c



a=[]
b=[]
c=[]
building_level={}
for j in train_data['manager_id'].values:
    building_level[j]=[0,0,0]
for j in range(train_data.shape[0]):
    temp=train_data.iloc[j]
    if temp['interest_level']=='low':
        building_level[temp['manager_id']][0]+=1
    if temp['interest_level']=='medium':
        building_level[temp['manager_id']][1]+=1
    if temp['interest_level']=='high':
        building_level[temp['manager_id']][2]+=1

for i in test_data['manager_id'].values:
    if i not in building_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(building_level[i][0]*1.0/sum(building_level[i]))
        b.append(building_level[i][1]*1.0/sum(building_level[i]))
        c.append(building_level[i][2]*1.0/sum(building_level[i]))
test_data['manager_level_low']=a
test_data['manager_level_medium']=b
test_data['manager_level_high']=c

manager_variable = ['manager_level_low','manager_level_medium','manager_level_high']