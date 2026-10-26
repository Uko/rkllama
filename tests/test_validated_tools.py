import json
from flask import jsonify
from test_validated_completion import native_route

TOOLS=[{'type':'function','function':{'name':'lookup','strict':True,'parameters':{'type':'object','properties':{'code':{'type':'string','enum':['A']}},'required':['code'],'additionalProperties':False}}}]

def test_required_tool_uses_validation_and_bounded_retry(native_route,monkeypatch):
 from rkllama.api.server_utils import ChatEndpointHandler
 calls=[]
 def generate(**kwargs):
  calls.append(kwargs)
  message={'role':'assistant','content':'No tool needed.'} if len(calls)==1 else {'role':'assistant','content':'','tool_calls':[{'id':'call_test','type':'function','function':{'name':'lookup','arguments':'{"code":"A"}'}}]}
  return jsonify({'id':'chatcmpl-test','object':'chat.completion','created':1,'model':'test','choices':[{'index':0,'message':message,'finish_reason':'tool_calls' if len(calls)>1 else 'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':3,'total_tokens':13}})
 monkeypatch.setattr(ChatEndpointHandler,'handle_request',generate)
 payload={'model':'test','messages':[{'role':'user','content':'Look up A'}],'tools':TOOLS,'tool_choice':'required','stream':False,'max_tokens':128}
 result=native_route.app.test_client().post('/v1/chat/completions',json=payload)
 assert result.status_code==200,result.get_data(as_text=True)
 assert len(calls)==2 and result.headers['X-RKLLama-Attempts']=='2'
 assert json.loads(result.get_json()['choices'][0]['message']['tool_calls'][0]['function']['arguments'])=={'code':'A'}
 assert result.get_json()['usage']['completion_tokens']==6
 assert payload['tools'][0]['function']['strict'] is True
