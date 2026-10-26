import json
from test_validated_completion import native_route,response,SCHEMA

def test_validated_stream_buffers_and_uses_native_aggregate_usage(native_route,monkeypatch):
 from rkllama.api.server_utils import ChatEndpointHandler
 calls=[]
 def generate(**kwargs):
  calls.append(kwargs)
  return response('{"answer":41}' if len(calls)==1 else '{"answer":42}')
 monkeypatch.setattr(ChatEndpointHandler,'handle_request',generate)
 payload={'model':'test','messages':[{'role':'user','content':'JSON please'}],'stream':True,'stream_options':{'include_usage':True},'response_format':{'type':'json_schema','json_schema':{'name':'answer','strict':True,'schema':SCHEMA}}}
 result=native_route.app.test_client().post('/v1/chat/completions',json=payload)
 assert result.status_code==200
 assert result.mimetype=='text/event-stream'
 raw=result.get_data(as_text=True);assert raw.endswith('data: [DONE]\n\n')
 events=[json.loads(l[6:]) for l in raw.splitlines() if l.startswith('data: ') and l!='data: [DONE]']
 assert len({e['id'] for e in events})==1
 content=''.join(c.get('delta',{}).get('content','') for e in events for c in e['choices'])
 assert json.loads(content)=={'answer':42}
 assert events[-1]['choices']==[] and events[-1]['usage']['completion_tokens']==6
 assert all(not c['stream'] for c in calls)
