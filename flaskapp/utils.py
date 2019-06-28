from flaskapp.packages import *

# text pre-process and keyword extraction
def pre_process(text, rm_stop=False):
    # lowercase
    text = text.lower()
    # remove special characters and digits
    text = re.sub("(\\W)+", ' ', text)
    # lemmatize (decided against stemming)
    words = text.split(' ')
    lem_words = []

    if rm_stop==True:
        for word in words:
            if (not word.isdigit() and (not word in stopwords)):
                lem_words.append(lemmer.lemmatize(word))
    else:
        for word in words:
            if not word.isdigit():
                lem_words.append(lemmer.lemmatize(word))

    cleaned_text = ' '.join(lem_words)
    return cleaned_text


# define function to call fasttext for inquiry classification
def get_group(custom_query):
    import subprocess
    result = subprocess.check_output(['./fastText/fasttext', 'predict-prob',
                                      './flaskapp/static/model/model_fasttext_cl_ova.bin',
                                      './flaskapp/static/data/tmp_query.txt', '-1', '0.1'], universal_newlines=True)
    label_idxs = [(n+9) for n in range(len(result)) if result.find('__label__', n)==n]
    label_list = []
    for idx in label_idxs:
        label_list.append([result[idx:(idx+4)], float(result[(idx+5):(idx+13)])])
    return label_list


def extract_keywords(coo_matrix, num_key, feature_names):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    # get feature name and tf-idf score for top n items
    if len(sorted_items) > num_key:
        sorted_items = sorted_items[:num_key]
    results = []

    for idx, tfidf in sorted_items:
        results.append([feature_names[idx], round(tfidf, 4)])
    return results


# synthesize search query components
def search_groups(label_list, num_groups=2):
    groups = min(len(label_list), num_groups)
    search_string = ''
    for i in range(groups):
        if search_string!='':
            search_string += '\n        OR '
        search_string += """substr(cpc.code, 1, 4) = '{}'""".format(label_list[i][0])
    return search_string

def keywords_comb(keywords, num_keywords=5):
    num_keywords = min(len(keywords), num_keywords)
    search_string = ''
    for i in range(num_keywords-1):
        for j in range(i+1, num_keywords):
            if (search_string!=''):
                search_string += '\n    OR '
            search_string += """(abstract.text LIKE '%{}%' AND abstract.text LIKE '%{}%')""".format(
                            keywords[i][0], keywords[j][0])
    return search_string


# define function to return search query
def search_query(label_list, keywords, limit=search_limit):

    query = """
    #standardSQL
    WITH P AS (
        SELECT
        DISTINCT publication_number,
        substr(cpc.code, 1,4) cpc4,
        floor(priority_date / 10000) priority_yr
        FROM `patents-public-data.patents.publications`,
        unnest(cpc) as cpc
        WHERE ({})
        AND floor(priority_date / 10000) >= 1999
    )

    SELECT
    STRING_AGG(assignee.name, ', ') as assignee_name,
    STRING_AGG(assignee.country_code, ', ') as applicant_country,
    P.priority_yr as patent_date,
    LOWER(patent_title.text) as title,
    STRING_AGG(abstract.text, ', ') as description
    FROM `patents-public-data.patents.publications` as pubs,
    UNNEST(abstract_localized) as abstract,
    UNNEST(assignee_harmonized) as assignee,
    UNNEST(title_localized) as patent_title
    JOIN P
      ON P.publication_number = pubs.publication_number
    WHERE abstract.language = 'en'
    AND patent_title.language = 'en'
    AND ({})
    GROUP BY patent_date, title

    LIMIT {}
    """.format(search_groups(label_list),
               keywords_comb(keywords),
               limit)

    return query


# query generation based on keywords of input and patent search

def googlebq_patents(label_list, keywords, topn=4):
    import google.auth
    from google.cloud import bigquery
    from google.cloud import bigquery_storage_v1beta1

    credentials, project_id = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    # Make clients.
    bqclient = bigquery.Client(credentials=credentials, project=project_id)
    bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient(credentials=credentials)

    query = search_query(label_list, keywords)

    df_search = (
        bqclient.query(query)
        .result()
        .to_dataframe(bqstorage_client=bqstorageclient)
    )
    df_search['patent_date'] = pd.to_numeric(df_search['patent_date'], errors='coerce').astype(np.int32)
    df_search.to_csv('./flaskapp/static/data/search_result.csv')
    key_query = keywords[0:4]

    return key_query, df_search


def local_patents(keywords):
    df_search = pd.read_csv('./flaskapp/static/data/search_result.csv', index_col=0)
    return keywords[0:4], df_search


# define cosine similarity calculation between query and search results
def cos_sim_df(feature_vecs):
    cos_sim_list = []
    for i in range(1, feature_vecs.shape[0]):
        if sum(feature_vecs[i]) == 0:
            cos_sim_list.append(0)
        else:
            cos_sim_list.append(1 - spatial.distance.cosine(feature_vecs[0], feature_vecs[i]))
    df_cos_sim = pd.DataFrame(cos_sim_list, columns=['cos_sim'])
    return df_cos_sim


def get_feature_vecs_simp(cus_keywords, candidate_list, fitted_vc, feature_names):
    new_feature_list = [kw_to_df(cus_keywords)]

    for patent in candidate_list:
        # compute IDF for search results
        tfidf_search = tfidf_transformer.fit_transform(fitted_vc.fit_transform([patent]))
        tmp_keywords = extract_keywords(tfidf_search.tocoo(), num_key, feature_names)

        df_keywords = kw_to_df(tmp_keywords)
        new_feature_list.append(df_keywords)

    df_feature_vec = pd.concat(new_feature_list, sort=False)
    df_feature_vec.drop(df_feature_vec.iloc[:, len(cus_keywords):], inplace=True, axis=1)

    df_feature_vec.reset_index(drop=True, inplace=True)
    df_feature_vec.fillna(0, inplace=True)
    feature_vecs = np.asarray(df_feature_vec.values.tolist())
    return feature_vecs


# define function to turn list keywords in dataframe
def kw_to_df(keywords):
    df_key = pd.DataFrame(keywords).transpose()
    new_header = df_key.iloc[0]
    df_key = df_key[1:]
    df_key.columns = new_header
    return df_key
