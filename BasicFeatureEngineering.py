num_vars = ['bathrooms','bedrooms','latitude','longitude','price']
cat_vars = ['building_id','manager_id','display_address','street_address', 'borocode','ntaname']
text_vars = ['description','features']
date_var = 'created'
image_var = 'photos'
id_var = 'listing_id'

# date variable processing
full_data['created_datetime'] = pd.to_datetime(full_data['created'], format="%Y-%m-%d %H:%M:%S")
# full_data['created_year']=full_data['created_datetime'].apply(lambda x:x.year) ## low variant
full_data['created_month']=full_data['created_datetime'].apply(lambda x:x.month)
full_data['created_day']=full_data['created_datetime'].apply(lambda x:x.day) 
full_data['created_dayofweek']=full_data['created_datetime'].apply(lambda x:x.dayofweek)
full_data['created_dayofyear']=full_data['created_datetime'].apply(lambda x:x.dayofyear)
full_data['created_weekofyear']=full_data['created_datetime'].apply(lambda x:x.weekofyear)
full_data['created_epoch']=full_data['created_datetime'].apply(lambda x:x.value//10**9)

#basic features from description and photos
full_data['rooms'] = full_data['bedrooms'] + full_data['bathrooms'] 
full_data['num_of_photos'] = full_data['photos'].apply(lambda x:len(x))
full_data['num_of_features'] = full_data['features'].apply(lambda x:len(x))
full_data['len_of_desc'] = full_data['description'].apply(lambda x:len(x)) # conghui cut
full_data['words_of_desc'] = full_data['description'].apply(lambda x:len(re.sub('['+string.punctuation+']', '', x).split()))


full_data['nums_of_desc'] = full_data['description']\
        .apply(lambda x:re.sub('['+string.punctuation+']', '', x).split())\
        .apply(lambda x: len([s for s in x if s.isdigit()])) 
        
full_data['has_phone'] = full_data['description'].apply(lambda x:re.sub('['+string.punctuation+']', '', x).split())\
        .apply(lambda x: [s for s in x if s.isdigit()])\
        .apply(lambda x: len([s for s in x if len(str(s))==10]))\
        .apply(lambda x: 1 if x>0 else 0)
full_data['has_email'] = full_data['description'].apply(lambda x: 1 if '@renthop.com' in x else 0)

additional_num_vars = ['rooms','num_of_photos','num_of_features','len_of_desc',
                    'words_of_desc','has_phone','has_email']


#This is a few ratio features with price variable
full_data['avg_word_len'] = full_data[['len_of_desc','words_of_desc']]\
                                    .apply(lambda x: x[0]/x[1] if x[1]!=0 else 0, axis=1)
    
full_data['price_per_room'] = full_data[['price','rooms']].apply(lambda x: x[0]/x[1] if x[1]!=0 else 0, axis=1)
full_data['price_per_bedroom'] = full_data[['price','bedrooms']].apply(lambda x: x[0]/x[1] if x[1]!=0 else 0, axis=1)
full_data['price_per_bathroom'] = full_data[['price','bathrooms']].apply(lambda x: x[0]/x[1] if x[1]!=0 else 0, axis=1)
full_data['price_per_photo'] = full_data[['price','num_of_photos']].apply(lambda x: x[0]/x[1] if x[1]!=0 else 0, axis=1) # conghui cut


full_data['photos_per_room'] = full_data[['num_of_photos','rooms']].apply(lambda x: x[0]/x[1] if x[1]!=0 else 0, axis=1) # conghui cut


interactive_num_vars = ['avg_word_len','price_per_room','price_per_bedroom','price_per_bathroom','price_per_photo',
                        'photos_per_room']