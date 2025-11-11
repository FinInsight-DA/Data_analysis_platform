import json
import requests
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime


class InsightPageAPI:
    """Insightpage API 클라이언트 클래스"""
    
    def __init__(self, token: str, uri: str = "http://fins.inpage.ai"):
        """
        API 클라이언트 초기화
        
        Parameters:
        -----------
        token : str
            API 인증 토큰
        uri : str, optional
            API 서버 URI (기본값: "http://fins.inpage.ai")
        """
        self.token = token
        self.uri = uri
        self.headers = {
            "Content-type": "application/json",
            "Accept": "text/plain"
        }
    
    def get_documents(
        self,
        start_date: str,
        end_date: str,
        keyword: str,
        synonyms: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, List[str]]] = None,
        category: Optional[List[str]] = None,
        category_sub: Optional[List[str]] = None,
        language: str = "ko",
        size: int = 10000,
        from_index: int = 1
    ) -> Dict[str, Any]:
        """
        뉴스 데이터 조회 API
        
        Parameters:
        -----------
        start_date : str
            시작 날짜 (형식: "YYYY-MM-DD")
        end_date : str
            종료 날짜 (형식: "YYYY-MM-DD")
        keyword : str
            검색 키워드
        synonyms : List[str], optional
            동의어 리스트
        filter_dict : Dict[str, List[str]], optional
            필터 조건 {"inOr": [], "inAnd": [], "exOr": [], "exAnd": []}
        category : List[str], optional
            카테고리 리스트
        category_sub : List[str], optional
            서브 카테고리 리스트
        language : str, optional
            언어 코드 (기본값: "ko")
        size : int, optional
            조회할 문서 수 (기본값: 10000)
        from_index : int, optional
            시작 인덱스 (기본값: 1)
        
        Returns:
        --------
        Dict[str, Any]
            API 응답 결과
        """
        if synonyms is None:
            synonyms = []
        
        if filter_dict is None:
            filter_dict = {
                "inOr": [],
                "inAnd": [],
                "exOr": [],
                "exAnd": []
            }
        
        data = {
            "startDate": start_date,
            "endDate": end_date,
            "search": {
                "keyword": keyword,
                "synonyms": synonyms,
                "filter": filter_dict
            },
            "language": language,
            "size": size,
            "from": from_index,
            "token": self.token
        }
        
        if category is not None:
            data["category"] = category
        
        if category_sub is not None:
            data["category_sub"] = category_sub
        
        response = requests.post(
            url=f"{self.uri}/api/biz/v1/documents",
            data=json.dumps(data),
            headers=self.headers
        )
        
        return response.json()
    
    def get_documents_to_csv(
        self,
        start_date: str,
        end_date: str,
        keyword: str,
        filename: str,
        synonyms: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, List[str]]] = None,
        category: Optional[List[str]] = None,
        category_sub: Optional[List[str]] = None,
        language: str = "ko",
        size: int = 10000,
        from_index: int = 1
    ) -> pd.DataFrame:
        """
        뉴스 데이터 조회 후 CSV 파일로 저장
        
        Parameters:
        -----------
        (get_documents와 동일한 파라미터)
        filename : str
            저장할 CSV 파일명
        
        Returns:
        --------
        pd.DataFrame
            문서 데이터프레임
        """
        result = self.get_documents(
            start_date=start_date,
            end_date=end_date,
            keyword=keyword,
            synonyms=synonyms,
            filter_dict=filter_dict,
            category=category,
            category_sub=category_sub,
            language=language,
            size=size,
            from_index=from_index
        )
        
        df = pd.DataFrame(result.get('documents', []))
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"저장 완료: {filename} ({len(df)}개 문서)")
        
        return df
    
    def get_documents_paginated(
        self,
        start_date: str,
        end_date: str,
        keyword: str,
        start_page: int,
        end_page: int,
        synonyms: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, List[str]]] = None,
        category: Optional[List[str]] = None,
        category_sub: Optional[List[str]] = None,
        language: str = "ko",
        size: int = 10000,
        save_csv: bool = True
    ) -> List[Dict[str, Any]]:
        """
        페이지네이션을 사용한 뉴스 데이터 조회
        
        Parameters:
        -----------
        start_page : int
            시작 페이지
        end_page : int
            종료 페이지
        save_csv : bool, optional
            CSV 파일 저장 여부 (기본값: True)
        (기타 파라미터는 get_documents와 동일)
        
        Returns:
        --------
        List[Dict[str, Any]]
            전체 결과 리스트
        """
        all_results = []
        
        for page in range(start_page, end_page + 1):
            print(f"페이지 {page} 조회 중...")
            
            result = self.get_documents(
                start_date=start_date,
                end_date=end_date,
                keyword=keyword,
                synonyms=synonyms,
                filter_dict=filter_dict,
                category=category,
                category_sub=category_sub,
                language=language,
                size=size,
                from_index=size * page + 1
            )
            
            all_results.append(result)
            
            if save_csv and 'documents' in result:
                filename = f"{keyword}_{page}.csv"
                df = pd.DataFrame(result['documents'])
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"  → {filename} 저장 완료 ({len(df)}개 문서)")
        
        return all_results
    
    def get_buzz(
        self,
        start_date: str,
        end_date: str,
        keyword: str,
        synonyms: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, List[str]]] = None,
        category: Optional[List[str]] = None,
        category_sub: Optional[List[str]] = None,
        language: str = "ko",
        interval: str = "day"
    ) -> List[Dict[str, Any]]:
        """
        언급량 분석 API
        
        Parameters:
        -----------
        start_date : str
            시작 날짜 (형식: "YYYY-MM-DD")
        end_date : str
            종료 날짜 (형식: "YYYY-MM-DD")
        keyword : str
            검색 키워드
        synonyms : List[str], optional
            동의어 리스트
        filter_dict : Dict[str, List[str]], optional
            필터 조건
        category : List[str], optional
            카테고리 리스트
        category_sub : List[str], optional
            서브 카테고리 리스트
        language : str, optional
            언어 코드 (기본값: "ko")
        interval : str, optional
            집계 간격 (기본값: "day")
        
        Returns:
        --------
        List[Dict[str, Any]]
            날짜별 언급량 리스트
        """
        if synonyms is None:
            synonyms = []
        
        if filter_dict is None:
            filter_dict = {
                "inOr": [],
                "inAnd": [],
                "exOr": [],
                "exAnd": []
            }
        
        data = {
            "startDate": start_date,
            "endDate": end_date,
            "search": {
                "keyword": keyword,
                "synonyms": synonyms,
                "filter": filter_dict
            },
            "language": language,
            "interval": interval,
            "token": self.token
        }
        
        if category is not None:
            data["category"] = category
        
        if category_sub is not None:
            data["category_sub"] = category_sub
        
        response = requests.post(
            url=f"{self.uri}/api/biz/v1/buzz",
            data=json.dumps(data),
            headers=self.headers
        )
        
        return response.json()
    
    def get_sentiment(
        self,
        start_date: str,
        end_date: str,
        keyword: str,
        model: str = "esg",
        synonyms: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, List[str]]] = None,
        category: Optional[List[str]] = None,
        category_sub: Optional[List[str]] = None,
        language: str = "ko",
        interval: str = "day"
    ) -> List[Dict[str, Any]]:
        """
        감성분석 API
        
        Parameters:
        -----------
        start_date : str
            시작 날짜 (형식: "YYYY-MM-DD")
        end_date : str
            종료 날짜 (형식: "YYYY-MM-DD")
        keyword : str
            검색 키워드
        model : str, optional
            감성분석 모델 (기본값: "esg")
        synonyms : List[str], optional
            동의어 리스트
        filter_dict : Dict[str, List[str]], optional
            필터 조건
        category : List[str], optional
            카테고리 리스트
        category_sub : List[str], optional
            서브 카테고리 리스트
        language : str, optional
            언어 코드 (기본값: "ko")
        interval : str, optional
            집계 간격 (기본값: "day")
        
        Returns:
        --------
        List[Dict[str, Any]]
            날짜별 감성 점수 리스트
        """
        if synonyms is None:
            synonyms = []
        
        if filter_dict is None:
            filter_dict = {
                "inOr": [],
                "inAnd": [],
                "exOr": [],
                "exAnd": []
            }
        
        data = {
            "startDate": start_date,
            "endDate": end_date,
            "search": {
                "keyword": keyword,
                "synonyms": synonyms,
                "filter": filter_dict
            },
            "language": language,
            "interval": interval,
            "model": model,
            "token": self.token
        }
        
        if category is not None:
            data["category"] = category
        
        if category_sub is not None:
            data["category_sub"] = category_sub
        
        response = requests.post(
            url=f"{self.uri}/api/biz/v1/sentiment",
            data=json.dumps(data),
            headers=self.headers
        )
        
        return response.json()


# 사용 예시
if __name__ == "__main__":
    # API 토큰 (만료기간: 2025년 6월 30일)
    TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0eXBlIjoiQVBJIEtleSAtIFB1YmxpYyIsImV4cCI6MTc2NzIyNTU5OS4wfQ.kCXxCuJOs8__wVJdJqkeFz893I30HW5ai-hM1i4zaqE"
    
    # API 클라이언트 생성
    api = InsightPageAPI(token=TOKEN)
    
    # 1. 뉴스 데이터 조회
    print("=" * 80)
    print("뉴스 데이터 조회 예시")
    print("=" * 80)
    
    result = api.get_documents(
        start_date="2024-01-01",
        end_date="2024-12-31",
        keyword="반도체",
        synonyms=["반도체", "HBM"],
        size=100,
        from_index=1
    )
    print(f"조회된 문서 수: {len(result.get('documents', []))}")
    
    # 2. 언급량 분석
    print("\n" + "=" * 80)
    print("언급량 분석 예시")
    print("=" * 80)
    
    buzz_result = api.get_buzz(
        start_date="2025-03-01",
        end_date="2025-03-31",
        keyword="반도체",
        synonyms=["반도체", "HBM"]
    )
    print(f"분석 기간: {len(buzz_result)}일")
    
    # 3. 감성분석
    print("\n" + "=" * 80)
    print("감성분석 예시")
    print("=" * 80)
    
    sentiment_result = api.get_sentiment(
        start_date="2025-03-01",
        end_date="2025-03-31",
        keyword="반도체",
        synonyms=["반도체", "HBM"],
        model="esg"
    )
    print(f"분석 기간: {len(sentiment_result)}일")
