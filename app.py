from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os

from RS import recommender

app = FastAPI(title="논문 추천 API")

# CORS 설정 (개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def startup():
    """서버 시작 시 모델 로드"""
    recommender.load()


@app.get("/")
def root():
    """메인 페이지"""
    return FileResponse("static/index.html")


@app.get("/api/search")
def search(q: str, top_k: int = 5):
    """
    논문 검색 & 추천
    예: /api/search?q=neural+networks&top_k=5
    """
    return recommender.search(q, top_k=top_k)


@app.get("/api/paper/{paper_id}")
def get_paper(paper_id: int):
    """
    특정 논문 상세 + 유사 논문
    예: /api/paper/42
    """
    return recommender.get_paper(paper_id)


@app.get("/api/categories")
def get_categories():
    """카테고리 목록"""
    return recommender.get_categories()


@app.get("/api/graph")
def get_graph(limit: int = 200):
    """시각화용 그래프 데이터"""
    papers = recommender.papers[:limit]
    with open("data/edges.json") as f:
        edges = json.load(f)

    # limit 범위 내 엣지만 필터링
    filtered_edges = [e for e in edges if e[0] < limit and e[1] < limit]

    return {
        "nodes": [{"id": p["id"], "label": p["category"]} for p in papers],
        "edges": filtered_edges[:300],
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 서버 시작: http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
