from typing import TypedDict,List,Dict,Optional
class FoodState(TypedDict):
    goal : str
    meals : str
    parser_output : list[Dict]
    analyzer_output : Dict
    goal_analysis_content: str
    report_content : str
    retry_count : int
    parser_retry_count: int
    error : Optional[str]
    profile:Dict     
    targets : Dict
