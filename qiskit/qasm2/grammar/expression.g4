grammar expression;

// parser

start : expr | expr (',' expr)* | <EOF> ;

expr : '-' expr     # UMINUS
   | expr mulop expr # MULOPGRP
   | expr addop expr # ADDOPGRP
   | Identifier mulop Identifier # IMULOPIGRP
   | Identifier mulop expr # IMULOPEGRP
   | expr mulop Identifier # EMULOPIGRP
   | Identifier addop Identifier # IADDOPIGRP
   | Identifier addop expr # IADDOPEGRP
   | expr addop Identifier # EADDOPIGRP
   | unaryop expr #UNARYOPGRP 
   | '(' expr ')' # PARENGRP
   | NUMBER      # NUMBERGRP
   ;

addop : '+' | '-' ;

mulop : '*' | '/' | '%' | '^' ;

unaryop: 'sin' | 'cos' | 'tan' | 'sqrt' | 'exp' | 'ln' ;

// lexer

NUMBER : ('0' .. '9') + ('.' ('0' .. '9') +)?
	| 'pi' | 'π' ;

WS : [ \r\n\t] + -> skip ;

// identifiers
fragment ValidUnicode : [\p{Lu}\p{Ll}\p{Lt}\p{Lm}\p{Lo}\p{Nl}] ; // valid unicode chars
fragment Letter : [A-Za-z] ;
Identifier : (ValidUnicode | Letter)+ ;