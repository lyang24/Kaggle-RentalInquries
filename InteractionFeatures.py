# the template is the same for all codes below
# 1. create a dataframe with summary statics
# 2. left join
# 3. calculate percentile

price_by_manager = full_data.groupby('manager_id')['price'].agg([np.min,np.max,np.median,np.mean]).reset_index()
price_by_manager.columns = ['manager_id','min_price_by_manager',
                            'max_price_by_manager','median_price_by_manager','mean_price_by_manager']
full_data = pd.merge(full_data,price_by_manager, how='left',on='manager_id')

created_epoch_by_manager = full_data.groupby('manager_id')['created_epoch'].agg([np.min,np.max,np.median,np.mean]).reset_index()
created_epoch_by_manager.columns = ['manager_id','min_created_epoch_by_manager',
                            'max_created_epoch_by_manager','median_created_epoch_by_manager','mean_created_epoch_by_manager']
full_data = pd.merge(full_data,created_epoch_by_manager, how='left',on='manager_id')


price_by_building = full_data.groupby('building_id')['price'].agg([np.min,np.max,np.median,np.mean]).reset_index()
price_by_building.columns = ['building_id','min_price_by_building',
                            'max_price_by_building','median_price_by_building','mean_price_by_building']
full_data = pd.merge(full_data,price_by_building, how='left',on='building_id')


created_epoch_by_building = full_data.groupby('building_id')['created_epoch'].agg([np.min,np.max,np.median,np.mean]).reset_index()
price_by_building.columns = ['building_id','min_created_epoch_by_building',
                            'max_created_epoch_by_building','median_created_epoch_by_building','mean_created_epoch_by_building']
full_data = pd.merge(full_data,price_by_building, how='left',on='building_id')

price_by_disp_addr = full_data.groupby('display_address')['price'].agg([np.min,np.max,np.median,np.mean]).reset_index()
price_by_disp_addr.columns = ['display_address','min_price_by_disp_addr',
                            'max_price_by_disp_addr','median_price_by_disp_addr','mean_price_by_disp_addr']
full_data = pd.merge(full_data,price_by_disp_addr, how='left',on='display_address')




full_data['price_percentile_by_manager']=\
            full_data[['price','min_price_by_manager','max_price_by_manager']]\
            .apply(lambda x:(x[0]-x[1])/(x[2]-x[1]) if (x[2]-x[1])!=0 else 0.5,
                  axis=1)
full_data['price_percentile_by_building']=\
            full_data[['price','min_price_by_building','max_price_by_building']]\
            .apply(lambda x:(x[0]-x[1])/(x[2]-x[1]) if (x[2]-x[1])!=0 else 0.5,
                  axis=1)
full_data['price_percentile_by_disp_addr']=\
            full_data[['price','min_price_by_disp_addr','max_price_by_disp_addr']]\
            .apply(lambda x:(x[0]-x[1])/(x[2]-x[1]) if (x[2]-x[1])!=0 else 0.5,
                  axis=1)


full_data['created_epoch_percentile_by_manager']=\
            full_data[['created_epoch','min_created_epoch_by_manager','max_created_epoch_by_manager']]\
            .apply(lambda x:(x[0]-x[1])/(x[2]-x[1]) if (x[2]-x[1])!=0 else 0.5,
                  axis=1)
        
price_by_borocode = full_data.groupby('borocode')['price'].agg([np.min,np.max,np.median,np.mean]).reset_index()
price_by_borocode.columns = ['borocode','min_price_by_borocode',
                            'max_price_by_borocode','median_price_by_borocode','mean_price_by_borocode']
full_data = pd.merge(full_data,price_by_borocode, how='left',on='borocode')

full_data['price_percentile_by_borocode']=\
            full_data[['price','min_price_by_borocode','max_price_by_borocode']]\
            .apply(lambda x:(x[0]-x[1])/(x[2]-x[1]) if (x[2]-x[1])!=0 else 0.5,
                  axis=1)
        
price_by_boro_bed = full_data.groupby('boro_bed')['price'].agg([np.min,np.max,np.median,np.mean]).reset_index()
price_by_boro_bed.columns = ['boro_bed','min_price_by_boro_bed',
                            'max_price_by_boro_bed','median_price_by_boro_bed','mean_price_by_boro_bed']
full_data = pd.merge(full_data,price_by_boro_bed, how='left',on='boro_bed')

full_data['price_percentile_by_boro_bed']=\
            full_data[['price','min_price_by_boro_bed','max_price_by_boro_bed']]\
            .apply(lambda x:(x[0]-x[1])/(x[2]-x[1]) if (x[2]-x[1])!=0 else 0.5,
                  axis=1)

price_by_cluster = full_data.groupby('clusters')['price'].agg([np.min,np.max,np.median,np.mean]).reset_index()
price_by_cluster.columns = ['clusters','min_price_by_clusters',
                            'max_price_by_clusters','median_price_by_clusters','mean_price_by_clusters']
full_data = pd.merge(full_data,price_by_cluster, how='left',on='clusters')

price_by_ntanew = full_data.groupby('ntanew')['price'].agg([np.min,np.max,np.median,np.mean]).reset_index()
price_by_ntanew.columns = ['ntanew','min_price_by_ntanew',
                            'max_price_by_ntanew','median_price_by_ntanew','mean_price_by_ntanew']
full_data = pd.merge(full_data,price_by_ntanew, how='left',on='ntanew')

full_data['price_percentile_by_ntanew']=\
            full_data[['price','min_price_by_ntanew','max_price_by_ntanew']]\
            .apply(lambda x:(x[0]-x[1])/(x[2]-x[1]) if (x[2]-x[1])!=0 else 0.5,
                  axis=1)

full_data['price_percentile_by_clusters']=\
            full_data[['price','min_price_by_clusters','max_price_by_clusters']]\
            .apply(lambda x:(x[0]-x[1])/(x[2]-x[1]) if (x[2]-x[1])!=0 else 0.5,
                  axis=1)
created_epoch_by_display_address = full_data.groupby('display_address')['created_epoch'].agg([np.min,np.max,np.median,np.mean]).reset_index()
created_epoch_by_display_address.columns = ['display_address','min_created_epoch_by_display_address',
                            'max_created_epoch_by_display_address','median_created_epoch_by_display_address','mean_created_epoch_by_display_address']
full_data = pd.merge(full_data,created_epoch_by_display_address, how='left',on='display_address')

created_dayofyear_by_created_month = full_data.groupby('created_month')['created_dayofyear'].agg([np.min,np.max,np.median,np.mean]).reset_index()
created_dayofyear_by_created_month.columns = ['created_month','min_created_dayofyear_by_created_month',
                            'max_created_dayofyear_by_created_month','median_price_by_created_dayofyear_month','mean_created_dayofyear_by_created_month']
full_data = pd.merge(full_data,created_dayofyear_by_created_month, how='left',on='created_month')

full_data['created_dayofyear_percentile_by_created_month']=\
            full_data[['created_dayofyear','min_created_dayofyear_by_created_month','max_created_dayofyear_by_created_month']]\
            .apply(lambda x:(x[0]-x[1])/(x[2]-x[1]) if (x[2]-x[1])!=0 else 0.5,
                  axis=1)

num_cat_vars = ['median_price_by_manager','mean_price_by_manager',
                'median_price_by_building','mean_price_by_building',
                'median_price_by_disp_addr','mean_price_by_disp_addr',
                'median_created_epoch_by_manager','mean_created_epoch_by_manager',
                'price_percentile_by_manager','price_percentile_by_building',
                'price_percentile_by_disp_addr','created_epoch_percentile_by_manager',
                'median_price_by_clusters', 'mean_price_by_clusters', 
                'price_percentile_by_clusters',
                'median_created_epoch_by_display_address',#'mean_created_epoch_by_display_address',
                'median_price_by_boro_bed','mean_price_by_boro_bed',
                #'median_price_by_ntanew',
                'mean_price_by_ntanew','price_percentile_by_ntanew',
                'created_dayofyear_percentile_by_created_month'
               ]