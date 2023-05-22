import uvicorn
from fastapi import FastAPI
import pandas as pd
from sentence_transformers import SentenceTransformer
import json
from sklearn.metrics.pairwise import cosine_similarity
import pickle


app=FastAPI()

@app.get('/')
def index():
     return {'message':'Welcome to MccDatamatchingApplication'}


@app.post('/Predict')
def get_data(str1):

    def readInvoice(str1):
      obj=json.loads(str1)
      invoice=obj["Invoice"]
      df1=pd.DataFrame.from_dict(invoice)
      df1['ItemDescription'].replace("[^a-zA-Z0-9]", " ", regex = True, inplace = True)
      df1['MatchedPOLineNumber']=''
      df1['Score']=''
      with open('E:\MccDataMatching\model_pickle','rb') as f:
        mp=pickle.load(f)
      df1['Embedding']=df1['ItemDescription'].apply(lambda x: mp.encode(str(x)))
      return df1

    invoiceDf=readInvoice(str1)
    print(invoiceDf)

    def readPo(str1):
      obj=json.loads(str1)
      po=obj["PO_num"]
      df2=pd.DataFrame.from_dict(po)
      df2['Short_Text'].replace("[^a-zA-Z0-9]", " ", regex = True, inplace = True)
      df2['Flag']=0
      df2['Score']=''
      with open('E:\MccDataMatching\model_pickle','rb') as f:
        mp=pickle.load(f)
      df2['Embedding']=df2['Short_Text'].apply(lambda x: mp.encode(str(x)))
      return df2

    poDf=readPo(str1)
    print(poDf)

    for i in invoiceDf.itertuples():
      temp_df = poDf[poDf['PONumber']==i.PONumber]
    
      for t in temp_df.itertuples():
          if t.Flag == 0:
            temp_df.at[t.Index, 'Score'] =float(cosine_similarity(i.Embedding.reshape(1,-1), t.Embedding.reshape(1,-1)))
            print(temp_df)

          if temp_df['Score'].empty:
            temp_df['Score']=0
                    
      temp_df['Score'] =temp_df['Score'].apply(pd.to_numeric, errors='ignore')
      max_index = temp_df['Score'].idxmax()
      poDf.iloc[[max_index], [6]] = 1
      invoiceDf.at[i.Index, 'MatchedPOLineNumber'] = int(temp_df.loc[[max_index], ["PoLine#"]].values)
      invoiceDf.at[i.Index, 'Score']=float(temp_df.loc[[max_index],["Score"]].values)
      temp_df.drop(columns=['PONumber', 'PoLine#','Short_Text','Embedding', 'Flag', 'Score'])

      print(invoiceDf)
      print(poDf)

    resultDf=invoiceDf[['IRN','PONumber','MatchedPOLineNumber','Score']]
    print(resultDf)
    dictobj=resultDf.to_dict(orient='index')
    print(dictobj)
    response=json.dumps(dictobj)
    return response

if __name__=='__MccPoLineIdentification__':
  uvicorn.run(app)





    

    

    
