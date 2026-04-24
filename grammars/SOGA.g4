grammar SOGA;

progr : (data ';')*? (instr ';' | array ';')*;

data : 'data' symvars '=' list;

array: 'array[' NUM ']' IDV;

instr : assignment | conditional | prune | observe | loop;

assignment: symvars '=' (const | add | poly_asgmt | trig_asgmt | exp_asgmt) | 'skip';

const: const_term (('+'|'-') const_term)*?;
const_term: (NUM | par | idd) ('*' (NUM | idd | par))?;
add: add_term (('+'|'-') add_term)*?;
add_term: ((NUM | idd | par) '*')? vars | const_term;

poly_asgmt: poly_aterm (('+' | '-')? poly_aterm)*?;
poly_aterm: (NUM | idd | par) '*' poly_mono | poly_mono | (NUM | idd | par);
poly_mono: poly_pfactor ('*' poly_pfactor)*;
poly_pfactor: vars ('^' NUM)?;

trig_asgmt: trig_aterm (('+' | '-')? trig_aterm)*?;
trig_aterm: (NUM | idd | par) '*' trig_mono | trig_mono | (NUM | idd | par);
trig_mono: trig_sfactor ('*' trig_sfactor)*;
trig_sfactor: vars ('^' NUM)?
            | ('cos(' vars ')' | 'sin(' vars ')') ('^' NUM)?;

exp_asgmt: exp_aterm (('+' | '-')? exp_aterm)*?;
exp_aterm: (NUM | idd | par) '*' exp_mono | exp_mono | (NUM | idd | par);
exp_mono: exp_sfactor ('*' exp_sfactor)*;
exp_sfactor: vars ('^' NUM)?
           | 'exp(' vars ')' ('^' NUM)?;

conditional: ifclause elseclause 'end if';

ifclause : 'if' bexpr '{' block '}';
elseclause : 'else' '{' block '}';
block : (instr ';')+;
bexpr : lexpr ('<'|'<='|'>='|'>') (NUM | idd | par) | symvars ('=='|'!=') (NUM | idd | par);
lexpr: monom (('+'|'-') monom)*?;
monom: ((NUM | idd | par) '*')? vars;

prune : 'prune(' NUM ')';

observe: 'observe(' bexpr ')';

loop : 'for' IDV 'in range(' (NUM | idd) ')' '{' block '}' 'end for';

vars: symvars | gm | uniform;
idd: IDV '[' (NUM | IDV) ']';
symvars : IDV | idd;
gm: 'gm(' list ',' list ',' list ')';
uniform: 'uniform(' list ',' NUM ')';
list: '[' (NUM | par) (',' (NUM | par))*? ']';

par: '_' IDV;

IDV : ALPHA (ALPHA|DIGIT)*;
NUM : '-'? DIGIT+ ('.' DIGIT*)?;

COMM : '/*' .*? '*/' -> skip;
WS : (' '|'\t'|'\r'|'\n') -> skip;

fragment
ALPHA : [a-zA-Z];
DIGIT : [0-9];
