#!/usr/bin/env python3
"""Process a shard of S2 remaining parquets. Reads file list from s2_remain_s{shard}.txt"""
import argparse,hashlib,io,json,os,shutil,tarfile,time
from pathlib import Path
os.environ.setdefault("HF_HOME","/mnt/bn/search-auto-eval-v2/zhouwei/hf_cache")
os.environ.setdefault("HF_TOKEN","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")

BN="/mnt/bn/search-auto-eval-v2/zhouwei/LightMGT/pt_data"
HDFS="/mnt/hdfs/weichow/maskedit/t2i-pt"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shard",type=int,required=True)
    a=p.parse_args()

    flist=f"{BN}/s2_remain_s{a.shard}.txt"
    with open(flist) as f: files=[l.strip() for l in f if l.strip()]
    print(f"Shard {a.shard}: {len(files)} parquets")

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from PIL import Image

    dl=f"{BN}/downloads/s2final_s{a.shard}"
    out=f"{BN}/output/s2final_s{a.shard}"
    os.makedirs(dl,exist_ok=True);os.makedirs(out,exist_ok=True);os.makedirs(HDFS,exist_ok=True)

    records=[];errors=0;t0=time.time()
    for i,fname in enumerate(files):
        try:
            tmp=hf_hub_download("common-canvas/commoncatalog-cc-by",fname,repo_type="dataset",cache_dir=dl+"/.c")
            local=dl+f"/p{i}.parquet";shutil.copy2(tmp,local)
            pf=pq.ParquetFile(local)
            cols={c.lower():c for c in pf.schema_arrow.names}
            img_col=cols.get("jpg")or cols.get("image")
            cap_col=cols.get("caption")or cols.get("blip2_caption")
            tar_path=out+f"/cc_f{a.shard}_{i}.tar"
            seen=set()
            with tarfile.open(tar_path,"w") as tf:
                for batch in pf.iter_batches(batch_size=500,columns=[img_col,cap_col]):
                    for j in range(len(batch)):
                        try:
                            v=batch.column(img_col)[j].as_py()
                            c=batch.column(cap_col)[j].as_py()
                            b=v.get("bytes") if isinstance(v,dict) else v
                            if not b or len(b)<100:errors+=1;continue
                            uid=hashlib.md5(b[:2048]).hexdigest()
                            if uid in seen:continue
                            seen.add(uid)
                            img=Image.open(io.BytesIO(b))
                            if img.mode!="RGB":img=img.convert("RGB")
                            w,h=img.size
                            if max(w,h)>1536:s=1536/max(w,h);img=img.resize((int(w*s),int(h*s)),Image.LANCZOS);w,h=img.size
                            buf=io.BytesIO();img.save(buf,format="JPEG",quality=95)
                            name=f"images/commoncatalog/{uid}.jpg"
                            info=tarfile.TarInfo(name=name);info.size=buf.tell();buf.seek(0)
                            tf.addfile(info,buf)
                            records.append({"uid":uid,"caption":c if isinstance(c,str) else "","image":name,"height":h,"width":w})
                        except:errors+=1
            os.remove(local)
            elapsed=time.time()-t0
            print(f"  [{i+1}/{len(files)}] {len(records)} records, {errors} err, {elapsed:.0f}s")
        except Exception as e:print(f"  FAIL {fname}: {e}")
    # JSON + upload
    idx=0
    for i in range(0,len(records),10000):
        chunk=records[i:i+10000]
        jname=f"commoncatalog_h2f{a.shard}_{idx:04d}.json"
        with open(out+"/"+jname,"w") as f:json.dump(chunk,f)
        shutil.copy2(out+"/"+jname,HDFS+"/"+jname)
        idx+=1
    print(f"DONE: {len(records)} records, {idx} JSONs, {errors} errors")
    shutil.rmtree(dl,ignore_errors=True);shutil.rmtree(out,ignore_errors=True)

if __name__=="__main__":main()
