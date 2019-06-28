from flaskapp.utils import *

def get_competitors(custom_query):
    custom_query = pre_process(custom_query)
    with open('./flaskapp/static/data/tmp_query.txt', 'w') as text_file:
        print(custom_query, file=text_file)

    label_list = get_group(custom_query)

    # import vocab (for vectorization)
    fitted_vc = CountVectorizer(vocabulary=pickle.load(open(
                './flaskapp/static/data/vocab_'+label_list[0][0]+'.pkl', 'rb')))
    feature_names = fitted_vc.get_feature_names()

    tfidf_query = tfidf_transformer.fit_transform(fitted_vc.fit_transform([custom_query]))
    cus_keywords = extract_keywords(tfidf_query.tocoo(), num_key, feature_names)
    key_query, df_search = googlebq_patents(label_list, cus_keywords)
    print('Search query for patents based on keywords extracted from input: {}'.format(key_query))
    candidate_list = df_search['description'].tolist()

    df_sim = cos_sim_df(get_feature_vecs_simp(cus_keywords, candidate_list, fitted_vc, feature_names))
    df_sim.sort_values(by='cos_sim', ascending=False, inplace=True)

    cmpt_list = []
    for idx in df_sim[0: search_limit].index.values:
        if df_sim.at[idx, 'cos_sim'] > 0:
            cmpt_list.append([df_search.at[idx, 'assignee_name'], df_search.at[idx, 'applicant_country'],
                              df_search.at[idx, 'patent_date'], df_search.at[idx, 'title'],
                              df_sim.at[idx, 'cos_sim']])
    df_cmpt = pd.DataFrame(cmpt_list, columns=['assignee_name', 'applicant_country',
                            'patent_date', 'title', 'cos_sim'])
    df_cmpt = list_cleanup(df_cmpt)

    return df_cmpt


# clean up returned list of companies
def list_cleanup(df_cmpt):
    df_cmpt.fillna(value={'applicant_country': ', ', 'assignee_name': ', '}, inplace=True)
    df_cmpt['applicant_country'] = df_cmpt['applicant_country'].str.split(', ').apply(
                                    lambda x: ', '.join(list(filter(None, list(set(x))))))
    df_cmpt['assignee_name'] = df_cmpt['assignee_name'].str.split(', ').apply(
                                lambda x: ', '.join(list(filter(None, list(set(x))))))
    df_cmpt['cos_sim'] = df_cmpt['cos_sim'].round(4)

    return df_cmpt
