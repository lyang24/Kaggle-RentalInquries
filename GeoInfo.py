# missing data handling via geocoder package
# beware of google's query limit
missingCoords = full_data[(full_data.longitude == 0) | (full_data.latitude == 0)]
missingGeoms = (missingCoords.street_address + ', New York').apply(geocoder.google)

full_data.loc[(full_data.longitude == 0) | (full_data.latitude == 0), 'latitude'] = missingGeoms.apply(lambda x: x.lat)
full_data.loc[(full_data.longitude == 0) | (full_data.latitude == 0), 'longitude'] = missingGeoms.apply(lambda x: x.lng)

missing_data = pd.DataFrame({'lat':missingGeoms.apply(lambda x: x.lat), 'long':missingGeoms.apply(lambda x: x.lng)})
missing_data.to_csv(r'..a_few_missing.csv')

# the code below was not implemented because of competition rules
# However, it is a great way to get the neighborhoods of the apartments base on coordinates
full_data['geometry'] = full_data.apply(lambda x: Point((float(x.longitude), float(x.latitude))), axis=1)
#geodata from NYC.gov
poly = gpd.GeoDataFrame.from_file(r'..nynta.geojson')
gdat = gpd.GeoDataFrame(full_data, crs = poly.crs, geometry='geometry')
geo_data = sjoin(gdat,poly, how='left', op='within')

geo_var = ['borocode','ntaname']
full_data = pd.merge(left = full_data, right = geo_data[geo_var],left_index=True,right_index=True, how = 'left')

# instead i went with the DB scan clustering method 
# summary static of clusters and other variables were used as features
from sklearn.cluster import DBSCAN
scale = preprocessing.StandardScaler()
cluster_vars = ['created_weekofyear','latitude', 'longitude','bedrooms'] + manager_variable
cluster_df = full_data[cluster_vars]
cluster_df['bedrooms'] = cluster_df['bedrooms'].clip_upper(5)
cluster_df['bedrooms'] = cluster_df['bedrooms'].map(lambda x: 0.7 * x)
cluster_df['created_weekofyear'] = cluster_df['created_weekofyear'].map(lambda x: 6 * x)
cluster_df[cluster_vars] = scale.fit_transform(cluster_df[cluster_vars])
db = DBSCAN(eps=0.3, min_samples=20).fit(cluster_df)
labels = db.labels_
full_data['clusters'] = labels


# calculate euclidean distance to city center 
# this is the a geofeature
from scipy.spatial import distance
geo = ['latitude', 'longitude']
full_data['coords'] = list(zip(full_data.latitude, full_data.longitude))
def dist_to_center(x):
    true_center = np.array((40.7128,-74.0059)).reshape(-1,2)
    k = distance.cdist(np.array(x).reshape(-1,2), true_center, 'euclidean') 
    return k.astype(float)
full_data['dist_to_center'] = full_data['coords'].map(dist_to_center)
full_data['dist_to_center'] = full_data['dist_to_center'].str[0]
full_data['dist_to_center'] = full_data['dist_to_center'].str[0]