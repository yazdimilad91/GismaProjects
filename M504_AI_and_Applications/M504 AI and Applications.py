import pandas as pd

RealState = pd.read_csv("https://drive.google.com/uc?export=download&id=12sKC1M4-G8G7sJM-QBh05HQRtqgeFPHr")

RealState.info()

print("Rows before:", len(RealState))
print("Nulls before:", RealState.isna().sum().sum())
print("Duplicates before:", RealState.duplicated().sum())

RealState["Unternehmenswert (Mio. €)"] = pd.to_numeric(RealState["Unternehmenswert (Mio. €)"], errors="coerce")
RealState = RealState.dropna()
RealState = RealState.drop_duplicates()


print("Rows after:", len(RealState))
print("Nulls after:", RealState.isna().sum().sum())
print("Duplicates after:", RealState.duplicated().sum())

RealState.to_csv("companies_150_de.csv", index=False)

RealState.head()


###embediing 

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


RealState = pd.read_csv("companies_150_de.csv")


def row_to_text_and_id_and_meta(row):
    meta = {col: str(val) if not pd.isna(val) else "" for col, val in row.items()}
    doc_text = " | ".join(f"{k}: {v}" for k, v in meta.items())
    row_id = str(row.name)
    return doc_text, row_id, meta


documents, ids, metadatas = [], [], []
for idx, r in RealState.iterrows():
    d, rid, m = row_to_text_and_id_and_meta(r)
    documents.append(d)
    ids.append(rid)
    metadatas.append(m)


client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("companies")
#model = SentenceTransformer("intfloat/multilingual-e5-large")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embs = model.encode(documents, convert_to_numpy=True)


collection.add(
    embeddings=[e.tolist() for e in embs],
    documents=documents,
    metadatas=metadatas,
    ids=ids
)


##### make main function 
import pandas as pd, numpy as np, chromadb, json
from sentence_transformers import SentenceTransformer
from groq import Groq
from geopy.geocoders import Nominatim
import osmnx as ox, matplotlib.pyplot as plt

groq_client = Groq(api_key="")
chroma = chromadb.PersistentClient(path="./chroma_db")
col = chroma.get_or_create_collection("companies")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def parse(question):
    sys = (
        "Map the user request to filters using ONLY:\n"
        "- Aktivitätstyp: [Projektentwickler,Immobiliengesellschaft,Betreiber,Sonstige]\n"
        "- Haupt-Assetklasse: [Wohnnutzung,Gewerbe,Pflegeheim]\n"
        "- Asset-Detail: [Studentenwohnen,Seniorenwohnen,Mehrfamilienhäuser,Einfamilienhäuser,"
        "Hotel,Serviced Apartments,Büro,Einzelhandel,Pflegeheim,Logistik]\n"
        "- Standort: city names\n"
        "- Unternehmenswert (Mio. €): numbers\n"
        "Return JSON: {filters:{Aktivitätstyp:[],Haupt-Assetklasse:[],Asset-Detail:[],Standort:[]},"
        "need_geo:false,geo_city:\"\",geo_distance_km:0,"
        "need_value_filter:false,value_min:null,value_max:null,need_plot_bar:false,"
        "need_semantic:false,semantic_query:\"\"}"
    )
    r = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"system","content":sys},{"role":"user","content":question}],
        response_format={"type":"json_object"},temperature=0,max_tokens=300
    )
    return json.loads(r.choices[0].message.content)
	
def nearby(city, dist_km):
    geo = Nominatim(user_agent="milad").geocode(city)
    if not geo: return []
    gdf = ox.features_from_point((geo.latitude, geo.longitude), tags={"place":True}, dist=dist_km*1000)
    return gdf[gdf["place"].isin(["town","village","city","hamlet"])]["name"].dropna().unique().tolist()
	
	
	
def ask(q, top_k=10):
    p = parse(q)
    got = col.get(include=["metadatas","embeddings"], limit=10000)
   RealState, embs = pd.DataFrame(got["metadatas"]), got["embeddings"]
    
    filters = p.get("filters") or {}

    for c in ["Aktivitätstyp", "Haupt-Assetklasse", "Asset-Detail", "Standort"]:
        vals = [str(v).strip().lower() for v in filters.get(c, [])]
        if not vals or c not in RealState.columns:
           continue
       RealState = RealState[
           RealState[c].astype(str).str.lower().apply(
               lambda s: any(v in s for v in vals)
           )
        ]

    if p["need_geo"]:
        cities = nearby(p["geo_city"], p["geo_distance_km"])
        if cities: 
            RealState = RealState[RealState["Standort"].isin(cities)]
            geo = Nominatim(user_agent="milad").geocode(p["geo_city"])
            gdf = ox.features_from_point((geo.latitude, geo.longitude), tags={"place": True}, dist=p["geo_distance_km"]*1000)
            ax = gdf[gdf["place"].isin(["town","village","city","hamlet"])].plot(figsize=(6,6), color="lightgray"); gdf[gdf["name"].isin(df["Standort"])].plot(ax=ax, color="red", markersize=30)

            
    if p["need_value_filter"]:
        if p["value_min"] is not None:
            RealState = RealState[RealState["Unternehmenswert (Mio. €)"].astype(float) >= p["value_min"]]
        if p["value_max"] is not None:
            RealState = RealState[RealState["Unternehmenswert (Mio. €)"].astype(float) <= p["value_max"]]
        if p["need_plot_bar"]:
            plt.bar(RealState["Unternehmen"], RealState["Unternehmenswert (Mio. €)"]); plt.xticks(rotation=90); plt.show()

    if p["need_semantic"]:
        q_emb = embedder.encode([p["semantic_query"] or q], convert_to_numpy=True)[0]
        mat = np.array(embs,dtype=float)
        sims = mat @ (q_emb/(np.linalg.norm(q_emb)+1e-8))
        RealState["_score"]=sims
        RealState = RealState.sort_values("_score",ascending=False).head(top_k).drop(columns=["_score"])
    else:
        RealState = RealState.head(top_k)
     
    return RealState[["Unternehmen","Unternehmensbeschreibung","Aktivitätstyp",
               "Haupt-Assetklasse","Asset-Detail","Standort","Unternehmenswert (Mio. €)"]]
			   
	
### Template_questions
ask("Hotels within 30 km of Berlin")

ask(" am looking for companies in the field of Genetic")

ask("Plot a bar chart of companies in Hamburg with Unternehmenswert between 150 and 500 million euros")

ask("List hotel companies in northeast germany")

ask("I am looking to buy some place in Essen that cost less than 50")