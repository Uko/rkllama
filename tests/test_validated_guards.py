import copy,json
import pytest
from flask import jsonify
from test_validated_completion import native_route,response,SCHEMA
from test_validated_tools import TOOLS

def payload(**extra):
 return {'model':'test','messages':[{'role':'user','content':'Return JSON'}],'stream':False,'response_format':{'type':'json_schema','json_schema':{'name':'answer','strict':True,'schema':SCHEMA}},**extra}

@pytest.mark.parametrize('text',['{"answer":41}','{"answer":true}','{"answer":42,"extra":1}','{"answer":42,"answer":42}','{"answer":NaN}','not JSON'])
@pytest.mark.parametrize('stream',[False,True])
def test_invalid_candidates_never_escape_and_attempts_are_bounded(native_route,monkeypatch,text,stream):
 from rkllama.api.server_utils import ChatEndpointHandler
 calls=[]
 def generate(**kw):calls.append(kw);return response(text)
 monkeypatch.setattr(ChatEndpointHandler,'handle_request',generate)
 result=native_route.app.test_client().post('/v1/chat/completions',json=payload(stream=stream))
 assert result.status_code==502 and len(calls)==2
 assert result.get_json()['error']['code']=='output_validation_failed'
 assert result.get_json()['usage']['completion_tokens']==6
 assert 'choices' not in result.get_json()

@pytest.mark.parametrize('schema',[{'$ref':'https://invalid.example/schema'},{'type':'string','pattern':'a+'},{'type':'string','format':'email'},{'type':42}])
def test_unsupported_or_invalid_schema_rejected_before_inference(native_route,monkeypatch,schema):
 from rkllama.api.server_utils import ChatEndpointHandler
 def forbidden(**kw):raise AssertionError('must not invoke native backend')
 monkeypatch.setattr(ChatEndpointHandler,'handle_request',forbidden)
 p=payload();p['response_format']['json_schema']['schema']=schema
 assert native_route.app.test_client().post('/v1/chat/completions',json=p).status_code==400

@pytest.mark.parametrize('extra',[{'unexpected':True},{'stream':True,'stream_options':{'include_usage':'yes'}},{'stream':False,'stream_options':{'include_usage':True}},{'max_tokens':1,'max_completion_tokens':2}])
def test_adaptation_does_not_hide_invalid_client_fields(native_route,monkeypatch,extra):
 from rkllama.api.server_utils import ChatEndpointHandler
 def forbidden(**kw):raise AssertionError('must not invoke native backend')
 monkeypatch.setattr(ChatEndpointHandler,'handle_request',forbidden)
 assert native_route.app.test_client().post('/v1/chat/completions',json=payload(**extra)).status_code==400

def tool_response(name='lookup',arguments='{"code":"A"}'):
 return jsonify({'id':'chatcmpl-test','object':'chat.completion','created':1,'model':'test','choices':[{'index':0,'message':{'role':'assistant','content':'','tool_calls':[{'id':'call_test','type':'function','function':{'name':name,'arguments':arguments}}]},'finish_reason':'tool_calls'}],'usage':{'prompt_tokens':10,'completion_tokens':3,'total_tokens':13}})

def test_tool_stream_has_indexed_validated_calls(native_route,monkeypatch):
 from rkllama.api.server_utils import ChatEndpointHandler
 monkeypatch.setattr(ChatEndpointHandler,'handle_request',lambda **kw:tool_response())
 p={'model':'test','messages':[{'role':'user','content':'Look up A'}],'stream':True,'tools':TOOLS,'tool_choice':'required'}
 r=native_route.app.test_client().post('/v1/chat/completions',json=p)
 assert r.status_code==200
 events=[json.loads(l[6:]) for l in r.get_data(as_text=True).splitlines() if l.startswith('data: ') and l!='data: [DONE]']
 call=events[0]['choices'][0]['delta']['tool_calls'][0]
 assert call['index']==0 and json.loads(call['function']['arguments'])=={'code':'A'}
 assert events[-1]['choices'][0]['finish_reason']=='tool_calls'

@pytest.mark.parametrize('arguments',['{"code":"B"}','{"code":"A","extra":1}','[]'])
def test_invalid_required_arguments_return_error(native_route,monkeypatch,arguments):
 from rkllama.api.server_utils import ChatEndpointHandler
 calls=[]
 def generate(**kw):calls.append(kw);return tool_response(arguments=arguments)
 monkeypatch.setattr(ChatEndpointHandler,'handle_request',generate)
 p={'model':'test','messages':[{'role':'user','content':'Look up A'}],'stream':False,'tools':TOOLS,'tool_choice':'required'}
 r=native_route.app.test_client().post('/v1/chat/completions',json=p)
 assert r.status_code==502 and len(calls)==2

def test_named_tool_restricts_offered_definitions(native_route,monkeypatch):
 from rkllama.api.server_utils import ChatEndpointHandler
 calls=[]
 def generate(**kw):calls.append(kw);return tool_response()
 monkeypatch.setattr(ChatEndpointHandler,'handle_request',generate)
 other=copy.deepcopy(TOOLS[0]);other['function']['name']='other'
 p={'model':'test','messages':[{'role':'user','content':'Look up A'}],'stream':False,'tools':TOOLS+[other],'tool_choice':{'type':'function','function':{'name':'lookup'}}}
 r=native_route.app.test_client().post('/v1/chat/completions',json=p)
 assert r.status_code==200 and len(calls[0]['tools'])==1
 assert calls[0]['tools'][0]['function']['name']=='lookup'

def test_strict_auto_tools_do_not_force_a_tool(native_route,monkeypatch):
 from rkllama.api.server_utils import ChatEndpointHandler
 monkeypatch.setattr(ChatEndpointHandler,'handle_request',lambda **kw:response('No tool is needed.'))
 p={'model':'test','messages':[{'role':'user','content':'Hello'}],'stream':False,'tools':TOOLS,'tool_choice':'auto'}
 r=native_route.app.test_client().post('/v1/chat/completions',json=p)
 assert r.status_code==200 and r.headers['X-RKLLama-Attempts']=='1'

def test_native_errors_do_not_trigger_another_generation(native_route,monkeypatch):
 from rkllama.api.server_utils import ChatEndpointHandler
 calls=[]
 def generate(**kw):calls.append(kw);return jsonify({'error':{'code':'native_inference_error'}}),500
 monkeypatch.setattr(ChatEndpointHandler,'handle_request',generate)
 r=native_route.app.test_client().post('/v1/chat/completions',json=payload())
 assert r.status_code==500 and len(calls)==1

@pytest.mark.parametrize('stream',[False,True])
def test_legacy_completion_schema_route(native_route,monkeypatch,stream):
 from rkllama.api.server_utils import GenerateEndpointHandler
 def generate(**kw):
  return jsonify({'id':'cmpl-test','object':'text_completion','created':1,'model':'test','choices':[{'index':0,'text':'{"answer":42}','finish_reason':'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':3,'total_tokens':13}})
 monkeypatch.setattr(GenerateEndpointHandler,'handle_request',generate)
 p=payload(stream=stream);p.pop('messages');p['prompt']='Return answer42'
 r=native_route.app.test_client().post('/v1/completions',json=p)
 assert r.status_code==200
 assert r.mimetype==('text/event-stream' if stream else 'application/json')
