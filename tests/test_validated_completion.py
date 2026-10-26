"""Post-generation contract tests. Model I/O is replaced; no native inference claimed."""
import json
import pytest
from flask import jsonify

SCHEMA={'type':'object','properties':{'answer':{'type':'integer','enum':[42]}},'required':['answer'],'additionalProperties':False}

@pytest.fixture
def native_route(monkeypatch):
 from rkllama.server import server
 monkeypatch.setattr(server,'is_gguf_model',lambda name:False)
 monkeypatch.setattr(server,'get_property_modelfile',lambda *args:'false')
 monkeypatch.setattr(server,'get_model_full_options',lambda name,path,options:options)
 monkeypatch.setattr(server.variables.worker_manager_rkllm,'exists_model_loaded',lambda name:True)
 return server

def response(content,completion=3):
 return jsonify({'id':'chatcmpl-test','object':'chat.completion','created':1,'model':'test','choices':[{'index':0,'message':{'role':'assistant','content':content},'finish_reason':'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':completion,'total_tokens':10+completion}})

def test_strict_json_validates_then_retries_without_leaking_bad_output(native_route,monkeypatch):
 from rkllama.api.server_utils import ChatEndpointHandler
 calls=[]
 def generate(**kwargs):
  calls.append(kwargs)
  return response('{"answer":41}' if len(calls)==1 else '{"answer":42}')
 monkeypatch.setattr(ChatEndpointHandler,'handle_request',generate)
 request={'model':'test','messages':[{'role':'user','content':'Return the answer as JSON.'}],'stream':False,'max_tokens':32,'response_format':{'type':'json_schema','json_schema':{'name':'answer','strict':True,'schema':SCHEMA}}}
 result=native_route.app.test_client().post('/v1/chat/completions',json=request)
 assert result.status_code==200,result.get_data(as_text=True)
 assert json.loads(result.get_json()['choices'][0]['message']['content'])=={'answer':42}
 assert len(calls)==2
 assert result.get_json()['usage']['completion_tokens']==6
 assert result.headers['X-RKLLama-Validation']=='post-generation'
 assert result.headers['X-RKLLama-Attempts']=='2'
 assert all(not call['stream'] for call in calls)
 assert request['messages']==[{'role':'user','content':'Return the answer as JSON.'}]
