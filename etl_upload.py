import os, glob
import pandas as pd
from supabase import create_client
from openai import OpenAI
from tiktoken import encoding_for_model
from docx import Document

oai = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
sp = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
enc = encoding_for_model("text-embedding-3-small")

KB_PATH = "knowledge_base/**"
for path in glob.glob(KB_PATH, recursive=True):
    if os.path.isdir(path): continue

    # 提取纯文本
    if path.endswith('.docx'):
        txt = "\n".join([p.text for p in Document(path).paragraphs])
    elif path.endswith('.csv'):
        txt = pd.read_csv(path).to_csv(index=False)
    else:
        with open(path, 'r', errors='ignore') as f:
            txt = f.read()

    # 拆分成 800 token
    tokens = enc.encode(txt)
    for i in range(0, len(tokens), 800):
        chunk_txt = enc.decode(tokens[i:i+800])
        emb = oai.embeddings.create(model="text-embedding-3-small", input=chunk_txt).data[0].embedding
        sp.table('kb_chunks').insert({
            "chunk": chunk_txt,
            "embedding": emb,
            "metadata": {"file": os.path.basename(path)}
        }).execute()
