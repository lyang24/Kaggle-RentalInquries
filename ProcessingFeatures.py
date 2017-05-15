full_data["cfeatures"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
cntvec = CountVectorizer(stop_words='english', max_features=200)
feature_sparse =cntvec.fit_transform(full_data["features"]\
                                     .apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x])))

feature_vars = ['cfeature_' + v for v in cntvec.vocabulary_]


cntvec = CountVectorizer(stop_words='english', max_features=100)
desc_sparse = cntvec.fit_transform(full_data["description"])
desc_vars = ['desc_' + v for v in cntvec.vocabulary_]

# basic embedding method on sklearn did not have time to try word2vec
