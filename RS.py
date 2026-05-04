import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity


class PaperRecommender:
    def __init__(self):
        self.embeddings = None
        self.papers = None
        self.loaded = False

    def load(self):
        """임베딩 & 메타데이터 로드"""
        self.embeddings = np.load("data/embeddings.npy")
        with open("data/papers.json", "r", encoding="utf-8") as f:
            self.papers = json.load(f)

        # 임베딩 정규화 (코사인 유사도 계산 최적화)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings_normalized = self.embeddings / (norms + 1e-8)
        self.loaded = True
        print(f"✅ 추천 시스템 로드 완료: {len(self.papers)}개 논문")

    def search(self, query: str, top_k: int = 5):
        """
        키워드로 논문 제목 검색 → 유사 논문 추천
        """
        if not self.loaded:
            self.load()

        query_lower = query.lower()

        # 1단계: 키워드 매칭으로 시드 논문 찾기
        seed_ids = [
            p["id"] for p in self.papers
            if query_lower in p["title"].lower()
            or query_lower in p["category"].lower()
        ]

        if not seed_ids:
            # 매칭 실패 시 카테고리 부분 매칭 시도
            categories = ["case based", "genetic", "neural", "probabilistic",
                          "reinforcement", "rule", "theory"]
            for cat in categories:
                if any(c in query_lower for c in cat.split()):
                    seed_ids = [p["id"] for p in self.papers if cat in p["category"].lower()][:3]
                    break

        if not seed_ids:
            return {"error": f"'{query}' 관련 논문을 찾을 수 없어요. 다른 키워드를 시도해보세요."}

        # 2단계: 시드 논문들의 평균 임베딩 계산
        seed_embeddings = self.embeddings_normalized[seed_ids]
        query_vec = seed_embeddings.mean(axis=0, keepdims=True)

        # 3단계: 코사인 유사도 계산
        similarities = cosine_similarity(query_vec, self.embeddings_normalized)[0]

        # 시드 논문 제외하고 상위 top_k 추출
        similarities[seed_ids] = -1
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            paper = self.papers[idx]
            results.append({
                "id": int(idx),
                "title": paper["title"],
                "category": paper["category"],
                "score": float(similarities[idx]),
            })

        return {
            "query": query,
            "seed_count": len(seed_ids),
            "results": results,
        }

    def get_paper(self, paper_id: int):
        """특정 논문 정보 + 유사 논문 반환"""
        if not self.loaded:
            self.load()

        if paper_id >= len(self.papers):
            return {"error": "논문을 찾을 수 없어요."}

        paper = self.papers[paper_id]
        similar = self.search(paper["category"], top_k=5)

        return {
            "paper": paper,
            "similar": similar.get("results", []),
        }

    def get_categories(self):
        """카테고리 목록 반환"""
        if not self.loaded:
            self.load()
        from collections import Counter
        counts = Counter(p["category"] for p in self.papers)
        return [{"name": k, "count": v} for k, v in counts.most_common()]


# 싱글톤 인스턴스
recommender = PaperRecommender()
