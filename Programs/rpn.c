#include <stdio.h>

#define END_OF_STRING '\0'
#define LIMIT 255

typedef enum {false, true} bool;

static long double buffer[LIMIT];
static unsigned char count = 0;

static long double calculate (char s[]);

static void push (long double num);
static long double pop (void);

static long double pow_ (long double num, int p);
static long double atof (char s[]);

static int isoperator (char c);
static int isdigit_ (char c);
static int isspace_ (char c);

int main (void) {
    char c, string[LIMIT];
    unsigned char index;
    for (index = 0; (c = getchar()) != '\n' &&
        index < LIMIT; index++)
        string[index] = c;
    printf("%.2Lf\n", calculate(string));
    return 0;
}

static long double calculate (char s[]) {
    long double operand;
    unsigned char i;
    
    char local_buffer[LIMIT];
    unsigned char local_count = 0;

    for (i = 0; s[i] != END_OF_STRING; i++) {
        if (isoperator(s[i]) && isdigit_(s[i + 1]) &&
            s[i + 1] != END_OF_STRING)
                local_buffer[local_count++] = '-';

        else if (isdigit_(s[i]) || s[i] == '.')
            local_buffer[local_count++] = s[i];

        else if (isspace_(s[i]) && local_count != 0) {
            local_buffer[local_count] = END_OF_STRING;
            push(atof(local_buffer));

            local_count = 0;
            local_buffer[0] = END_OF_STRING;
            
        } else if (isoperator(s[i])) {
            operand = pop();
            switch (s[i]) {
                case '+': push(pop() + operand); break;
                case '-': push(pop() - operand); break;
                case '*': push(pop() * operand); break;
                case '^': push(pow_(pop(), operand)); break;

                case '/':
                if (operand == 0.0) {
                    printf("division by zero\n");
                    return 0.0;
                }
                push(pop() / operand); break;

                case '|':
                if (operand == 0.0) {
                    printf("division by zero\n");
                    return 0.0;
                }
                push((long long) pop() / (long long) operand); 
                break;

                case '%':
                if (operand == 0) {
                    printf("division by zero\n");
                    return 0;
                }
                push((long long) pop() % (long long) operand); 
                break;
            }
        }
    }
    return buffer[0];
}

static void push (long double num) {
    if (count != LIMIT)
        buffer[count++] = num;
    else
        printf("Buffer is full\n");
}

static long double pop (void) {
    if (count != 0)
        return buffer[--count];
    else {
        printf("Buffer is void\n");
        return 0.0;
    }
}

static long double pow_ (long double num, int p) {
    if (p > 0)
        while (p-- > 1)
            num *= num;
    else if (p < 0) {
        num = 1 / num;
        while (p++ < -1)
            num *= num;
    }
    else return 1;
    return num;
}

static long double atof (char s[]) {
    long double val, power;
    int i, sign;
    for (i = 0; isspace_(s[i]); i++);
    sign = (s[i] == '-') ? -1 : 1;
    if (s[i] == '+' || s[i] == '-')
        i++;
    for (val = 0.0; isdigit_(s[i]); i++)
        val = 10.0 * val + (s[i] - '0');
    if (s[i] == '.')
        i++;
    for (power = 1.0; isdigit_(s[i]); i++) {
        val = 10.0 * val + (s[i] - '0');
        power *= 10;
    }
    return sign * val / power;
}

static int isoperator (char c) {
    switch (c) {
        case '+': case '-': case '*': case '/': 
        case '%': case '^': case '|': return 1;
        default: return 0;
    }
}

static int isdigit_ (char c) {
    if (c >= '0' && c <= '9')
        return 1;
    else return 0;
}

static int isspace_ (char c) {
    switch (c) {
        case ' ': case '\n': case '\t': return 1;
        default: return 0;
    }
}
