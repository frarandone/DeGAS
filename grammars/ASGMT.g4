grammar ASGMT;

assignment: symvars '=' (add | poly_sum | trig_sum | exp_sum);

add: add_term (('+')? add_term)*?;
add_term: (term '*')? term;
term: NUM | sub? par | sub? symvars | sub? gm;

poly_sum: poly_term (('+')? poly_term)*?;
poly_term: sub? (NUM | par | idd) '*' poly_mono
         | sub? poly_mono
         | sub? (NUM | par | idd);
poly_mono: poly_factor ('*' poly_factor)*;
poly_factor: (symvars | gm) ('^' NUM)?;

trig_sum: trig_term (('+')? trig_term)*?;
trig_term: sub? (NUM | par | idd) '*' trig_mono
         | sub? trig_mono
         | sub? (NUM | par | idd);
trig_mono: trig_factor ('*' trig_factor)*;
trig_factor: (symvars | gm) ('^' NUM)?
           | ('cos(' (symvars | gm) ')' | 'sin(' (symvars | gm) ')') ('^' NUM)?;

exp_sum: exp_term (('+')? exp_term)*?;
exp_term: sub? (NUM | par | idd) '*' exp_mono
        | sub? exp_mono
        | sub? (NUM | par | idd);
exp_mono: exp_factor ('*' exp_factor)*;
exp_factor: (symvars | gm) ('^' NUM)?
          | 'exp(' (symvars | gm) ')' ('^' NUM)?;

symvars : IDV | idd;
idd : IDV '[' (NUM | IDV) ']';
gm: 'gm(' list ',' list ',' list ')';
list: '[' (NUM | par) (',' (NUM | par))*? ']';

sub: '-';

par: '_' IDV;

IDV : ALPHA (ALPHA|DIGIT)*;
NUM : ('-')? DIGIT+ ('.' DIGIT*)?;

COMM : '/*' .*? '*/' -> skip;
WS : (' '|'\t'|'\r'|'\n') -> skip;

fragment
ALPHA : [a-zA-Z];
DIGIT : [0-9];
