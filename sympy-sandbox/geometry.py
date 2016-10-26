from sympy.geometry import intersection
# http://docs.sympy.org/latest/modules/geometry/utils.html

from sympy.geometry import Point, Segment
# http://docs.sympy.org/latest/modules/geometry/entities.html
# http://docs.sympy.org/latest/modules/geometry/points.html


class Vertex():
    def __init__(self, coord, edge):
        self.coordinates = coord # type: Point // 二次元平面上の座標
        self.incidentEdge = edge # type: HalfEdge // この点を始点とするエッジどれかひとつ（えらくずぼらな

class Face():
    def __init__(self, outer, inners):
        self.outerComponent = outer # type: HalfEdge|null; // この面を囲む右回り（内側）の片辺（矢印）のうちのどれかひとつ
        self.innerComponents = inners # type: HalfEdge[]; // この面の中に存在する複数の面の左回り（外側）の片辺（矢印）のうちのどれかひとつ

class HalfEdge():
    def __init__(self, origin, twin, incidentFace, next, prev):
        self.origin = origin # type: : Vertex; // 始点
        self.twin = twin # type: : HalfEdge; // 双対
        self.incidentFace = incidentFace # type: Face; // 矢印の左側の面（つまりこの辺は左回り
        self.next = next # type: HalfEdge; // 次の辺
        self.prev = prev # type: HalfEdge; // 前の辺

class Region():
    def __init__(self, face, verts, edges):
        self.face = face
        self.vertexes = verts
        self.halfEdges = edges

def createRectPointsFromLTAndRB(lt, rb):
    (x1, y1) = lt
    (x2, y2) = rb
    _lt = Point(x1, y1)
    _rt = Point(x2, y1)
    _rb = Point(x2, y2)
    _lb = Point(x1, y2)
    return (_lt, _rt, _rb, _lb)

def createRegion(pts):
    face = Face(None, []) # この世界には与えられた点群を順番に結んでできた領域しかないので外にも中にも何もない
    verts = [Vertex(pt, None) for pt in pts] # 接続辺は不明だがとりあえず点群を作る
    edges = []
    twins = []
    for i, vert in enumerate(verts):
        edge = HalfEdge(vert, None, face, None, None) # 次の辺と前の辺はとりあえず未定のまま
        twin = HalfEdge(verts[(i+1)%len(verts)], None, face, None, None)
        edge.twin = twin # 双対辺の決定
        twin.twin = edge
        vert.incidentEdge = edge # 接続辺が決定
        edges.append(edge)
        twins.append(twin)
    for i, (edge, twin) in enumerate(zip(edges, twins)): # 次の辺と前の辺を決める
        edge.next = edges[(i+1)%len(edges)]
        edge.prev = edges[(i-1)%len(edges)]
        twin.next = twins[(i-1)%len(twins)] # 双対なので edges とは符合が逆になっている
        twin.prev = twins[(i+1)%len(twins)]
    return Region(face, verts, edges) # edge の twin からアクセスできるので region には含まない


if __name__ == "__main__":
    rect = ((0,0), (1,1))
    pts = createRectPointsFromLTAndRB(*rect)
    print(createRegion(pts))