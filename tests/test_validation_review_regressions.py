import json
from types import SimpleNamespace
import pytest
from flask import jsonify
from test_validated_completion import native_route,response
from test_validated_guards import payload
from test_validated_tools import TOOLS


def test_applicator_explosion_rejected_without_running_validation():
 from rkllama.api.validated_completion import checked_schema
 from rkllama.api.request_options import RequestOptionError
 s={'properties':{'a':{}}}
 for _ in range(12):s={'allOf':[s],'unevaluatedProperties':False}
 with pytest.raises(RequestOptionError):checked_schema(s,'response_format')


@pytest.mark.parametrize('raw',[
 '<tool_call>\n<function=lookup>\n<parameter=code>B</parameter>\n</function>\n</tool_call>',
 '<tool_call>{"name":"lookup","arguments":{"code":"B","code":"A"}}</tool_call>',
 '{"name":"lookup","arguments":{"code":"B","code":"A"}}',
 '<tool_call>{"name":"unknown","arguments":{"code":"A"}}</tool_call>',
 '<tool_call>{"name":"lookup","arguments":{"code":"A"}}',
 '{"name":"lookup","arguments":{"code":"A"}} {"name":"lookup","arguments":{"code":"B","code":"A"}}',
])
def test_rejected_raw_tool_candidates_cannot_be_successful_auto_prose(native_route,monkeypatch,raw):
 from rkllama.api.server_utils import EndpointHandler,variables
 tokenizer=SimpleNamespace(chat_template='test',apply_chat_template=lambda *a,**kw:'rendered')
 class Tokenizer:
  chat_template='test'
  def apply_chat_template(self,*a,**kw):return 'rendered'
  def __call__(self,*a,**kw):return {'input_ids':[1,2]}
 monkeypatch.setattr(EndpointHandler,'get_tokenizer',lambda name:Tokenizer())
 calls=[]
 def inference(*args):
  calls.append(args);tokens=iter([raw,('finished',10,3,100,200,'stop')])
  return SimpleNamespace(poll=lambda timeout:True,recv=lambda:next(tokens),close=lambda:None)
 monkeypatch.setattr(variables.worker_manager_rkllm,'inference',inference)
 p={'model':'test','messages':[{'role':'user','content':'Hello'}],'stream':False,'tools':TOOLS,'tool_choice':'auto'}
 r=native_route.app.test_client().post('/v1/chat/completions',json=p)
 assert r.status_code==502,r.get_data(as_text=True)
 assert len(calls)==2 and 'choices' not in r.get_json()


def test_native_error_retains_known_prior_usage(native_route,monkeypatch):
 from rkllama.api.server_utils import ChatEndpointHandler
 calls=[]
 def generate(**kw):
  calls.append(kw)
  if len(calls)==1:return response('{"answer":41}')
  return jsonify({'error':{'code':'native_inference_error'}}),500
 monkeypatch.setattr(ChatEndpointHandler,'handle_request',generate)
 r=native_route.app.test_client().post('/v1/chat/completions',json=payload())
 assert r.status_code==500 and len(calls)==2
 assert r.get_json()['error']['code']=='native_inference_error'
 assert r.get_json()['usage']=={'prompt_tokens':10,'completion_tokens':3,'total_tokens':13}
 assert r.get_json()['usage_incomplete'] is True
