#!/usr/bin/env python3
"""Full offline inventory. Structural checks are not semantic certification."""
import argparse,json,re
from pathlib import Path
from html.parser import HTMLParser
from collections import Counter
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
NS="{http://www.w3.org/2000/svg}"
class Page(HTMLParser):
    def __init__(self):
        super().__init__();self.tags=[];self.ids=[];self.images=[];self.math=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs);self.tags.append((tag,a))
        if "id" in a:self.ids.append(a["id"])
        if tag=="img":self.images.append(a.get("src",""))
        if tag=="math":self.math.append(a.get("aria-label",""))
def inspect():
    manifest=json.loads((ROOT/"assets/course-graph.json").read_text())
    chapters=manifest["chapters"];expected=list(range(1,len(chapters)+1))
    errors=[];warnings=[];rows=[]
    if [x["n"] for x in chapters]!=expected:errors.append("graph IDs")
    if sorted(n for g in manifest["groups"] for n in g["chapters"])!=expected:errors.append("domain partition")
    by_group={n:g["id"] for g in manifest["groups"] for n in g["chapters"]}
    js=(ROOT/"assets/js/site.js").read_text()
    if json.loads(re.search(r"const courseGraph = (.*);",js).group(1))!=manifest:errors.append("embedded graph differs")
    idx=Page();idx.feed((ROOT/"index.html").read_text())
    cards=[a for t,a in idx.tags if t=="article" and "data-category" in a]
    if len(cards)!=len(chapters) or {a.get("data-category"):a.get("data-domain") for a in cards}!={c["tag"]:c["group"] for c in chapters}:errors.append("catalog mismatch")
    for c in chapters:
        n=c["n"];name=f"{n:02}";raw=(ROOT/(name+".html")).read_text();page=Page();page.feed(raw)
        if c["group"]!=by_group.get(n) or len(set(c["prerequisites"]))!=len(c["prerequisites"]) or any(x not in expected or x>=n for x in c["prerequisites"]):errors.append(f"{name}: invalid dependency")
        dup=[x for x,k in Counter(page.ids).items() if k>1]
        if dup:errors.append(f"{name}: duplicate IDs {dup}")
        formulas=re.findall(r"<math\b[^>]*>[\s\S]*?</math>",raw)
        if not formulas or len(formulas)!=len(page.math) or not all(page.math):errors.append(f"{name}: formula inventory")
        for i,formula in enumerate(formulas):
            try:
                m=ET.fromstring(formula)
                for child in m.iter():
                    tag=child.tag.split("}")[-1]
                    arity={"mfrac":2,"msub":2,"msup":2,"msubsup":3,"munder":2,"mover":2,"munderover":3}.get(tag)
                    if arity is not None and len(child)!=arity:errors.append(f"{name}: formula {i+1} {tag} arity")
            except ET.ParseError as exc:warnings.append(f"{name}: MathML XML parse: {exc}")
        e=json.loads((ROOT/"evidence"/(name+".json")).read_text());rev=e["source_revision"]
        if not re.fullmatch(r"[a-f0-9]{40}",rev):errors.append(f"{name}: invalid SHA")
        pinned=0;moving=[];other=[]
        for claim in e["claims"]:
            url=claim["source_url"]
            if "github.com/sgl-project/sglang/blob/" in url or "github.com/sgl-project/sglang/tree/" in url:
                found=re.search(r"/(?:blob|tree)/([a-f0-9]{40})/",url)
                if found and found.group(1)==rev:pinned+=1
                elif found:warnings.append(f"{name}: source revision differs {url}")
                else:moving.append(url)
            else:other.append(url)
        if moving:warnings.append(f"{name}: moving SGLang links {moving}")
        if pinned==0:warnings.append(f"{name}: no same-revision SGLang code link")
        labs=sorted(set(re.findall(r"examples/[a-z_]+\.py",raw)))
        for lab in labs:
            if not (ROOT/lab).is_file():errors.append(f"{name}: missing lab {lab}")
        rows.append({"chapter":name,"title":c["title"],"domain":c["group"],"prerequisites":c["prerequisites"],"formula_labels":page.math,"formula_count":len(formulas),"figures":page.images,"labs":labs,"source_revision":rev,"claims":len(e["claims"]),"same_revision_links":pinned,"moving_links":moving,"other_source_count":len(other),"semantic_review":"requires claim-level review"})
    svg_rows=[]
    for path in sorted((ROOT/"assets/diagrams").glob("*.svg")):
        svg=ET.parse(path).getroot();texts=svg.findall(".//"+NS+"text")
        for node in svg.findall(".//"+NS+"path"):
            if node.get("marker-end") and re.fullmatch(r"\s*[Mm]\s*(?:[-.0-9, ]+)?\s*",node.get("d","")):errors.append(f"{path.name}: degenerate arrow")
        svg_rows.append({"file":path.name,"text_nodes":len(texts),"viewBox":svg.get("viewBox"),"geometry_review":"browser svg-geometry.json"})
    used={image for row in rows for image in row["figures"]}
    all_svg={"assets/diagrams/"+s["file"] for s in svg_rows}
    if used!=all_svg:errors.append(f"SVG ownership: unused={sorted(all_svg-used)} missing={sorted(used-all_svg)}")
    return {"chapters":rows,"svg":svg_rows,"totals":{"chapters":len(rows),"math":sum(x["formula_count"] for x in rows),"svg":len(svg_rows),"claims":sum(x["claims"] for x in rows),"dependency_edges":sum(len(x["prerequisites"]) for x in rows),"domains":len(manifest["groups"])},"errors":errors,"warnings":warnings}
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",default="reports/quality-audit.json");args=p.parse_args()
    result=inspect();out=ROOT/args.out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result["totals"],ensure_ascii=False));print("errors",len(result["errors"]),"warnings",len(result["warnings"]))
    if result["errors"]:print("\n".join(result["errors"]));raise SystemExit(1)
if __name__=="__main__":main()
