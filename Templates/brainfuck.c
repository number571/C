#include <stdio.h>

// > "Hello World!"
// bf code example: https://en.wikipedia.org/wiki/Brainfuck
static const char *Code = \
	"++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>"\
	"---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.";

static void brainfuck(const char *code);

int main(void) {
	brainfuck(Code);
    return 0;
}

static void brainfuck(const char *code) {
	char memory[30000] = {0};
    char *ptr = memory;
    for (size_t i = 0; code[i] != '\0'; ++i) {
    	switch(code[i]) {
    		case '>':
    			++ptr;
    		break;
    		case '<':
    			--ptr;
    		break;
    		case '+':
    			++*ptr;
    		break;
    		case '-':
    			--*ptr;
    		break;
    		case '.':
    			putchar(*ptr);
    		break;
    		case ',':
    			*ptr = getchar();
    		break;
    		case '[': {
    			if (*ptr) {
    				continue;
    			}
    			size_t nesting = 1;
    			while(nesting) {
    				++i;
    				switch(code[i]) {
    					case '[':
    						++nesting;
    					break;
    					case ']':
    						--nesting;
    					break;
    				}
    			}
    		}
    		break;
    		case ']': {
    			if (!*ptr) {
    				continue;
    			}
    			size_t nesting = 1;
    			while(nesting) {
    				--i;
    				switch(code[i]) {
    					case '[':
    						--nesting;
    					break;
    					case ']':
    						++nesting;
    					break;
    				}
    			}
    		}
    		break;
    		
    	}
    }
}
