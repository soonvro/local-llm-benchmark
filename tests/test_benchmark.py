import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark import evaluate, parse_json_object

class BenchmarkTests(unittest.TestCase):
    def test_json_fence(self):
        self.assertEqual(parse_json_object('```json\n{"requirements": []}\n```'),{'requirements':[]})
    def test_evidence_correct(self):
        c={'jd':'R1 필수: Java 개발 경험.','resume':'Java로 앱을 개발했다.',
           'expected':{'R1':{'level':'required','status':'have'}}}
        response=json.dumps({'requirements':[{'id':'R1','level':'required','user_status':'have',
            'resume_quote':'Java로 앱을 개발했다.','jd_quote':'Java 개발 경험'}]},ensure_ascii=False)
        e=evaluate(c,response)
        self.assertEqual((e['status_correct'],e['evidence_valid'],e['false_have']),(1,1,0))
    def test_false_have_with_no_evidence(self):
        c={'jd':'R1 필수: Java 개발 경험.','resume':'Java 강의 신청함.',
           'expected':{'R1':{'level':'required','status':'unknown'}}}
        response=json.dumps({'requirements':[{'id':'R1','level':'required','user_status':'have',
            'resume_quote':'','jd_quote':'Java 개발 경험'}]},ensure_ascii=False)
        e=evaluate(c,response)
        self.assertEqual((e['status_correct'],e['evidence_valid'],e['false_have']),(0,0,1))
    def test_unknown_quote_empty(self):
        c={'jd':'R1 필수: 자격증 보유.','resume':'자격증 공부 중이다.',
           'expected':{'R1':{'level':'required','status':'unknown'}}}
        response=json.dumps({'requirements':[{'id':'R1','level':'required','user_status':'unknown',
            'resume_quote':'','jd_quote':'자격증 보유'}]},ensure_ascii=False)
        e=evaluate(c,response)
        self.assertEqual((e['status_correct'],e['evidence_valid']),(1,1))

if __name__=='__main__':unittest.main()
