from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
PLOTS = HERE.parent / "plots"
PDF = HERE / "spring_2026_is_poster.pdf"
PNG = HERE / "spring_2026_is_poster_preview.png"
DPI = 150
W, H = 36 * DPI, 24 * DPI

C = {
    "paper": "#e8e1cd", "header": "#79a9d8", "tan": "#c29d7f",
    "rose": "#bf7779", "blue": "#7eadda", "violet": "#7d7cdb",
    "green": "#8fb88f", "ink": "#050505", "white": "#ffffff",
}

def f(size, bold=False):
    p = "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
    return ImageFont.truetype(p, size) if Path(p).exists() else ImageFont.load_default()

TITLE, SUB = f(78, True), f(32, True)
H2, BODY, BODYB, SMALL, SMALLB, TINY = f(58, True), f(29), f(29, True), f(23), f(23, True), f(18, True)

def inch(x): return int(round(x * DPI))
def tw(d, s, font): return d.textbbox((0, 0), s, font=font)[2]
def th(d, s, font): return d.textbbox((0, 0), s, font=font)[3]

def wrap(d, text, font, width):
    out = []
    for para in text.split("\n"):
        words, cur = para.split(), ""
        for word in words:
            trial = (cur + " " + word).strip()
            if not cur or tw(d, trial, font) <= width:
                cur = trial
            else:
                out.append(cur); cur = word
        if cur: out.append(cur)
        out.append("")
    return out[:-1]

def text(d, x, y, s, font, width, gap=7):
    for line in wrap(d, s, font, width):
        if line:
            d.text((x, y), line, font=font, fill=C["ink"])
        y += th(d, "Ag", font) + gap
    return y

def panel(d, x, y, w, h, color, title):
    d.rounded_rectangle((x, y, x+w, y+h), radius=inch(.12), fill=color)
    d.text((x+inch(.22), y+inch(.22)), title, font=H2, fill=C["ink"])
    return x+inch(.25), y+inch(.88), w-inch(.5)

def image(canvas, path, x, y, w, h, cap=""):
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle((x, y, x+w, y+h), radius=inch(.035), fill=C["white"], outline="#5b4c40", width=2)
    im = Image.open(path).convert("RGB")
    pad, caph = inch(.06), inch(.22) if cap else 0
    im = ImageOps.contain(im, (w-2*pad, h-2*pad-caph), Image.Resampling.LANCZOS)
    canvas.paste(im, (x+(w-im.width)//2, y+pad))
    if cap:
        d.text((x+(w-tw(d, cap, TINY))//2, y+h-caph+inch(.02)), cap, font=TINY, fill="#222222")

def placeholder(d, x, y, w, h, name, note):
    d.rounded_rectangle((x, y, x+w, y+h), radius=inch(.04), fill=C["white"], outline="#5b4c40", width=3)
    d.text((x+(w-tw(d, name, SMALLB))//2, y+inch(.28)), name, font=SMALLB, fill=C["ink"])
    d.text((x+(w-tw(d, note, TINY))//2, y+inch(.66)), note, font=TINY, fill="#333333")

def bullets(d, x, y, items, width, font=BODY):
    for item in items:
        d.text((x, y), "-", font=font, fill=C["ink"])
        y = text(d, x+inch(.18), y, item, font, width-inch(.18), 4) + 2
    return y

canvas = Image.new("RGB", (W, H), C["paper"])
d = ImageDraw.Draw(canvas)
m = inch(.28)
d.rounded_rectangle((m, m, W-m, inch(2.36)), radius=inch(.14), fill=C["header"])
title = "Applied Spectral Graph Theory: Single-Cell Manifold Learning & Network Connectivity"
d.text(((W-tw(d,title,TITLE))//2, inch(.82)), title, font=TITLE, fill=C["ink"])
sub = "Spring 2026 Independent Study: Anmol Singh Josan"
d.text(((W-tw(d,sub,SUB))//2, inch(1.73)), sub, font=SUB, fill=C["ink"])

col1, col2, col3, top = inch(.28), inch(12), inch(24.2), inch(2.58)
cw1, cw2, cw3 = inch(11.42), inch(11.90), inch(11.52)

x,y,w = panel(d,col1,top,cw1,inch(5.55),C["tan"],"Introduction")
y = text(d,x,y,"Single-cell RNA sequencing measures thousands of genes across thousands of cells. The challenge is that biology is not arranged in neat rows and columns: it lives on a high-dimensional shape, or manifold, where nearby cells often share identity, state, or trajectory.\nMy independent study asked how spectral graph theory can recover that hidden shape. I treated cells as nodes, connected similar cells with weighted edges, and used the graph Laplacian to transform raw expression into a map of structure.",BODY,w)
d.rounded_rectangle((x,y+6,x+w,y+inch(.55)),radius=inch(.04),fill="#ffffff")
text(d,x+inch(.1),y+inch(.13),"Big idea: graph eigenvectors can turn a cloud of measurements into interpretable communities.",SMALLB,w-inch(.2),4)
placeholder(d,x,top+inch(4.05),inch(4.95),inch(1.2),"Add Image 1","manual graph or Laplacian")
placeholder(d,x+inch(5.35),top+inch(4.05),inch(4.95),inch(1.2),"Add Image 2","NetworkX or heat map")

x,y,w = panel(d,col1,inch(8.32),cw1,inch(5.72),C["rose"],"The Core Mathematics")
y = text(d,x,y,"To study a graph with linear algebra, I built three matrices:",BODY,w)
y = bullets(d,x,y,["Adjacency matrix A: stores which nodes are connected, and by how strongly.","Degree matrix D: records how much total edge weight touches each node.","Laplacian matrix L = D - A: acts like a discrete curvature or diffusion operator."],w)
text(d,x,y+6,"The eigenproblem L v = lambda v reveals the graph's natural modes. Small eigenvalues describe broad, smooth structure; their eigenvectors provide coordinates for clustering without assuming the groups in advance.\nFor biological data, normalized Laplacian ideas help keep high-degree cells from dominating the geometry unfairly.",BODY,w)

x,y,w = panel(d,col1,inch(14.23),cw1,inch(3.45),C["green"],"Why It Matters")
text(d,x,y,"Spectral methods connect pure linear algebra, network science, and biological discovery. A single eigenvector can identify a bottleneck in a social network, a weak point in infrastructure, or a boundary between cell states.\nIn medicine, this matters because hidden immune-cell communities may help explain why some patients respond to treatment and others do not.",BODY,w)

x,y,w = panel(d,col1,inch(17.87),cw1,inch(5.57),C["tan"],"Applications")
bullets(d,x,y,["Finding communities in large networks without labels.","Measuring robustness through algebraic connectivity.","Reducing high-dimensional biological data while preserving local neighborhoods.","Linking computational clusters to clinical response hypotheses."],w)
image(canvas,PLOTS/"dots_response_enrichment_volcano.png",x,inch(21),inch(5.05),inch(1.85),"Clinical response enrichment")
image(canvas,PLOTS/"dots_patient_composition_by_cluster.png",x+inch(5.35),inch(21),inch(5.05),inch(1.85),"Patient composition")

x,y,w = panel(d,col2,top,cw2,inch(5.1),C["violet"],"The Fiedler Value & Robustness")
text(d,x,y,"The smallest eigenvalue of a connected graph Laplacian is always lambda_1 = 0. The second smallest eigenvalue, lambda_2, is the Fiedler value, also called algebraic connectivity.\nIf lambda_2 > 0, the graph is connected. If it is close to zero, the graph has a narrow bridge or bottleneck. Cheeger's inequality connects this eigenvalue to the graph's minimum cut, so connectivity becomes measurable rather than just visual.",BODY,w)
placeholder(d,x,top+inch(3.68),inch(5.3),inch(1.1),"Add Image 3","targeted attack curve")
placeholder(d,x+inch(5.65),top+inch(3.68),inch(5.3),inch(1.1),"Add Image 4","spectral gap diagnostic")

x,y,w = panel(d,col2,inch(7.88),cw2,inch(9.02),C["blue"],"Case Study: Spectral Clustering")
y = text(d,x,y,"I applied the pipeline to a breast-cancer immunotherapy case study. Each cell was embedded in gene-expression space, connected to its nearest transcriptional neighbors, and clustered using eigenvectors of the graph Laplacian.",BODY,w)
for i,(val,lab) in enumerate([("2,500","cells analyzed"),("5","spectral communities"),("2","response views")]):
    sx=x+i*inch(3.66); sy=y+4
    d.rounded_rectangle((sx,sy,sx+inch(3.35),sy+inch(.55)),radius=inch(.04),fill="#ffffff")
    d.text((sx+(inch(3.35)-tw(d,val,SMALLB))//2,sy+6),val,font=SMALLB,fill=C["ink"])
    d.text((sx+(inch(3.35)-tw(d,lab,TINY))//2,sy+inch(.30)),lab,font=TINY,fill=C["ink"])
image(canvas,PLOTS/"dots_umap_clusters_and_response.png",x,inch(10.05),inch(11.05),inch(3.10),"UMAP: spectral clusters and clinical response")
image(canvas,PLOTS/"dots_tsne_clusters_and_response.png",x,inch(13.38),inch(11.05),inch(3.10),"t-SNE: stability check across another nonlinear projection")

x,y,w = panel(d,col2,inch(17.1),cw2,inch(6.34),C["tan"],"What I Learned")
text(d,x,y,"This project changed how I think about data. A dataset is not just a table; it can be a landscape. Spectral graph theory gives a way to ask where that landscape bends, where it pinches, and where it separates into meaningful regions.\nI also learned that beautiful math is not enough by itself. The pipeline needed preprocessing, graph construction choices, diagnostic checks, visualization, and interpretation. The satisfying part was watching those pieces lock together into a tool that could explain structure in real biological data.",BODY,w)

x,y,w = panel(d,col3,top,cw3,inch(5.45),C["tan"],"Results & Interpretation")
y = text(d,x,y,"The spectral clustering pipeline isolated distinct cell communities across patient cohorts. The clusters were not assigned by hand; they emerged from graph connectivity.\nIn the Week 9 readout, one T-cell-dominant community appeared response-associated, while a myeloid-dominant community appeared resistance-associated. That is the exciting bridge: an eigenvector calculation became a biological hypothesis about immune response.",BODY,w)
bullets(d,x,y,["Response signal: T-cell cluster, fold ratio 1.45, p = 7.24e-05.","Resistance signal: myeloid cluster, fold ratio 0.60, p = 1.39e-04."],w)

x,y,w = panel(d,col3,inch(8.23),cw3,inch(4.22),C["blue"],"More Figures to Add")
y=text(d,x,y,"This panel is the easiest place to swap in supporting visuals without disturbing the main story.",BODY,w)
bullets(d,x,y,["Full clinical cluster importance bar chart.","Cell-type composition or patient share chart."],w,SMALL)
image(canvas,HERE/"poster_clinical_cluster_importance.png",x,inch(11.02),inch(5.15),inch(1.05),"Cluster importance")
image(canvas,HERE/"poster_clinical_composition.png",x+inch(5.45),inch(11.02),inch(5.15),inch(1.05),"Clinical composition")

x,y,w = panel(d,col3,inch(12.66),cw3,inch(6.32),C["rose"],"The Research Journey")
y=text(d,x,y,"This project began with small graphs and hand calculations. I manually built adjacency, degree, and Laplacian matrices so the code would not feel like a black box.\nThe hardest shift was conceptual: moving from geometric intuition in ordinary space to structure in thousands of dimensions. The breakthrough came when the abstract math produced visible communities in the biological data.",BODY,w)
for i,(head,body) in enumerate([("Weeks 1-2","Matrix foundations and manual Laplacian derivations."),("Weeks 3-4","Fiedler values, centrality, heat maps, and robustness."),("Weeks 6-8","Stress tests, heat kernels, and higher-order topology ideas."),("Week 9","Single-cell case study and poster-ready analysis.")]):
    bx=x+i*inch(2.65); by=inch(17.32)
    d.rounded_rectangle((bx,by,bx+inch(2.45),by+inch(.92)),radius=inch(.04),fill="#ffffff")
    d.text((bx+10,by+8),head,font=TINY,fill=C["ink"])
    text(d,bx+10,by+34,body,f(15),inch(2.2),2)

x,y,w = panel(d,col3,inch(19.18),cw3,inch(4.26),C["green"],"Next Steps")
text(d,x,y,"Future work could test the pipeline on more real datasets, compare spectral clustering with graph neural networks, and extend from graph Laplacians to Hodge Laplacians for modeling multi-way cell interactions.",BODY,w)
text(d,x,inch(22.4),"Acknowledgement: thank you to my Independent Study advisor/content expert for helping shape the scope, math focus, and presentation feedback.",SMALLB,w,4)

canvas.save(PNG)
canvas.save(PDF, "PDF", resolution=DPI)
print(PDF)
print(PNG)
