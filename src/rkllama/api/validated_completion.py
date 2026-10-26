"""Bounded post-generation validation, not native constrained decoding.

Candidates are buffered; only schema-valid successes are returned. No tool is
executed here. Ordinary requests continue through the existing native handlers.
"""
import copy
import json
from flask import jsonify,make_response,Response,g,has_request_context
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from .request_options import RequestOptionError,validate_request_options,validate_tool_schema,strict_json

MAX_ATTEMPTS=2

def needs_validation(payload):
    value=payload.get('response_format');choice=payload.get('tool_choice')
    tools=payload.get('tools')
    strict_tools=isinstance(tools,list) and any(isinstance(t,dict) and isinstance(t.get('function'),dict) and t['function'].get('strict') is True for t in tools)
    return (isinstance(value,dict) and value.get('type')=='json_schema') or choice=='required' or isinstance(choice,dict) or strict_tools

def checked_schema(schema,param):
    if not isinstance(schema,dict):
        raise RequestOptionError(param,'Schema must be an object')
    try:
        if len(json.dumps(schema,allow_nan=False))>32768:raise ValueError('Schema exceeds 32 KiB')
        nodes=0
        def visit(value,depth=0):
            nonlocal nodes
            nodes+=1
            if depth>32 or nodes>2048:raise ValueError('Schema is too complex')
            if isinstance(value,dict):
                if set(value)&{'$ref','$dynamicRef','$recursiveRef','pattern','patternProperties','format'}:
                    raise ValueError('References, regular expressions and format assertions are unsupported')
                if '$schema' in value and value['$schema']!='https://json-schema.org/draft/2020-12/schema':
                    raise ValueError('Only JSON Schema draft 2020-12 is supported')
                for child in value.values():visit(child,depth+1)
            elif isinstance(value,list):
                for child in value:visit(child,depth+1)
        visit(schema)
        allowed={'type','properties','required','additionalProperties','items','enum','const','minimum','maximum','exclusiveMinimum','exclusiveMaximum','multipleOf','minLength','maxLength','minItems','maxItems','minProperties','maxProperties','title','description','default','$schema'}
        def check_keywords(node):
            if isinstance(node,bool):return
            if not isinstance(node,dict) or set(node)-allowed:raise ValueError('Unsupported schema keyword; combinators, unevaluated constraints and uniqueness are unavailable')
            for child in node.get('properties',{}).values():check_keywords(child)
            for key in ('items','additionalProperties'):
                if isinstance(node.get(key),dict):check_keywords(node[key])
        check_keywords(schema)
        Draft202012Validator.check_schema(schema)
    except (ValueError,RecursionError) as exc:
        raise RequestOptionError(param,str(exc)) from exc
    except Exception as exc:
        raise RequestOptionError(param,'Invalid JSON Schema') from exc
    return Draft202012Validator(schema)


def validated_stream(result,payload,is_chat):
    # Validation happens before HTTP headers/content are committed. This is
    # intentionally buffered SSE, not incremental native-token streaming.
    choice=result['choices'][0];include=payload.get('stream_options',{}).get('include_usage',False)
    base={k:result[k] for k in ('id','created','model')}
    base['object']='chat.completion.chunk' if is_chat else 'text_completion'
    part={'index':0,'finish_reason':None}
    if is_chat:
        part['delta']=copy.deepcopy(choice['message'])
        for index,call in enumerate(part['delta'].get('tool_calls',[])):call['index']=index
    else:part['text']=choice['text']
    final={'index':0,'finish_reason':choice.get('finish_reason','stop')}
    final['delta' if is_chat else 'text']={} if is_chat else ''
    def chunks():
        for selected in (part,final):
            event={**base,'choices':[selected]}
            if include:event['usage']=None
            yield 'data: '+json.dumps(event,ensure_ascii=False)+'\n\n'
        if include:yield 'data: '+json.dumps({**base,'choices':[],'usage':result['usage']})+'\n\n'
        yield 'data: [DONE]\n\n'
    return Response(chunks(),mimetype='text/event-stream')

def tool_plan(payload,inner):
    tools=payload.get('tools');choice=payload.get('tool_choice','auto')
    if not isinstance(tools,list) or not tools:
        raise RequestOptionError('tool_choice' if tools is None else 'tools','Validated tool requests require nonempty function definitions')
    validators={};clean=copy.deepcopy(tools)
    for index,tool in enumerate(clean):
        if not isinstance(tool,dict) or tool.get('type')!='function' or not isinstance(tool.get('function'),dict):
            raise RequestOptionError('tools','Only function tool definitions are supported')
        fn=tool['function'];name=fn.get('name')
        if not isinstance(name,str) or not name or name in validators:
            raise RequestOptionError('tools','Function names must be nonempty and unique')
        if 'strict' in fn and not isinstance(fn['strict'],bool):
            raise RequestOptionError('tools','strict must be boolean')
        fn.pop('strict',None)
        schema=fn.get('parameters',{'type':'object'})
        validate_tool_schema(schema,'tools.'+str(index)+'.function.parameters')
        validators[name]=checked_schema(schema,'tools.'+str(index)+'.function.parameters')
    selected=None
    if isinstance(choice,dict):
        if set(choice)!={'type','function'} or choice.get('type')!='function' or not isinstance(choice.get('function'),dict) or set(choice['function'])!={'name'}:
            raise RequestOptionError('tool_choice','Expected a named function choice')
        selected=choice['function']['name']
        if not isinstance(selected,str) or selected not in validators:
            raise RequestOptionError('tool_choice','Named function must be declared')
        clean=[t for t in clean if t['function']['name']==selected]
    elif choice not in ('auto','none','required'):
        raise RequestOptionError('tool_choice','Unsupported tool choice')
    inner['tools']=clean;inner['tool_choice']='none' if choice=='none' else 'auto'
    required=choice=='required' or selected is not None
    instruction=('You must return at least one function tool call. Do not answer in prose. ' if required else 'If a tool is called, its arguments must satisfy its schema. ')
    if selected:instruction+='Call only '+selected+'.'
    return validators,required,selected,instruction


def valid_tools(result,validators,required,selected):
    calls=result['choices'][0]['message'].get('tool_calls',[])
    raw=getattr(g,'rkllama_raw_candidate',None) if has_request_context() else None
    if raw is not None:
        from .format_utils import get_tool_calls,looks_like_tool_attempt
        definitions=[{'type':'function','function':{'name':name,'parameters':validator.schema}} for name,validator in validators.items()]
        parsed=get_tool_calls(raw,definitions)
        if (looks_like_tool_attempt(raw) and not parsed) or len(parsed)!=len(calls):
            raise ValueError('Rejected raw tool attempt')
    if required and not calls:raise ValueError('Required tool call missing')
    for call in calls:
        fn=call['function'];name=fn['name']
        if name not in validators or (selected is not None and name!=selected):raise ValueError('Undeclared or unselected tool')
        args=strict_json(fn['arguments'])
        if not isinstance(args,dict):raise ValueError('Tool arguments must be an object')
        validators[name].validate(args)


def validated_completion(payload,generate,*,is_chat=True):
    inner=copy.deepcopy(payload);value=payload.get('response_format');schema_mode=isinstance(value,dict) and value.get('type')=='json_schema'
    validator=None;validators={};required=False;selected=None
    if schema_mode:
        if payload.get('tools'):
            raise RequestOptionError('response_format','Combining schema output and tools is unsupported')
        spec=value.get('json_schema')
        if not isinstance(spec,dict) or not isinstance(spec.get('name'),str) or not spec['name']:
            raise RequestOptionError('response_format','Expected schema name and schema object')
        if 'strict' in spec and not isinstance(spec['strict'],bool):
            raise RequestOptionError('response_format.json_schema.strict','strict must be boolean')
        validator=checked_schema(spec.get('schema'),'response_format.json_schema.schema')
        inner['response_format']={'type':'json_object'}
        instruction='Return only one JSON value matching this schema, with no markdown or commentary: '+json.dumps(spec['schema'],ensure_ascii=False)
    else:
        if not is_chat:raise RequestOptionError('tool_choice' if 'tool_choice' in payload else 'tools','Tools require the chat endpoint')
        validators,required,selected,instruction=tool_plan(payload,inner)
    # Validate original stream options and numeric fields before adapting them.
    validate_request_options(inner,openai=True,chat=is_chat)
    inner['stream']=False;inner.pop('stream_options',None)
    usage={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}
    for attempt in range(1,MAX_ATTEMPTS+1):
        if is_chat:inner.setdefault('messages',[]).append({'role':'user','content':instruction})
        else:inner['prompt']=inner.get('prompt','')+'\n'+instruction
        previous=getattr(g,'rkllama_validation_active',False)
        g.rkllama_validation_active=True;g.rkllama_raw_candidate=None
        try:response=make_response(generate(copy.deepcopy(inner)))
        finally:g.rkllama_validation_active=previous
        response.headers['X-RKLLama-Validation']='post-generation';response.headers['X-RKLLama-Attempts']=str(attempt)
        if response.status_code!=200:
            error=response.get_json(silent=True)
            if isinstance(error,dict) and any(usage.values()):
                known=dict(usage)
                for key in known:known[key]+=(error.get('usage') or {}).get(key,0)
                error['usage']=known;error['usage_incomplete']=True
                response.set_data(json.dumps(error))
            return response
        result=response.get_json()
        if not isinstance(result,dict):raise RuntimeError('Native completion was not a JSON object')
        for key in usage:usage[key]+=(result.get('usage') or {}).get(key,0)
        try:
            if schema_mode:
                value=strict_json(result['choices'][0]['message']['content'] if is_chat else result['choices'][0]['text'])
                validator.validate(value)
            else:valid_tools(result,validators,required,selected)
        except (ValueError,KeyError,TypeError,IndexError,RecursionError,ValidationError):
            instruction='The previous response failed validation. Return a valid response satisfying the preceding schema/tool instructions.'
            continue
        if schema_mode:
            field=result['choices'][0]['message'] if is_chat else result['choices'][0]
            field['content' if is_chat else 'text']=json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)
        result['usage']=usage
        response=validated_stream(result,payload,is_chat) if payload.get('stream') else jsonify(result)
        response.headers['X-RKLLama-Validation']='post-generation';response.headers['X-RKLLama-Attempts']=str(attempt)
        return response
    response=jsonify({'error':{'type':'server_error','code':'output_validation_failed','message':'No validated completion within the two-attempt limit'},'usage':usage})
    response.status_code=502;response.headers['X-RKLLama-Validation']='post-generation';response.headers['X-RKLLama-Attempts']=str(MAX_ATTEMPTS)
    return response
