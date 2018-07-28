#include <stdio.h>
#define LIMIT 255

double buffer[LIMIT];
unsigned char count = 0;

double calculate (char s[]);

void push (double num);
double pop (void);

double atof (char s[]);

int isoperate (char c);
int isdigit_ (char c);
int isspace_ (char c);

int main (void) {
    char c, string[LIMIT]; // 1 2 - 4 5 + *
    unsigned char index;
    for (index = 0; (c = getchar()) != '\n' &&
        index < LIMIT; index++)
        string[index] = c;
    printf("%.2lf\n", calculate(string));
    return 0;
}

double calculate (char s[]) {
    double operand;
    unsigned char i;
    
    char local_buffer[LIMIT];
    unsigned char local_count = 0;

    for (i = 0; s[i] != '\0'; i++) {
        if (isdigit_(s[i]) || s[i] == '.')
            local_buffer[local_count++] = s[i];

        else if (isspace_(s[i]) && local_count != 0) {
            local_buffer[local_count] = '\0';
            push(atof(local_buffer));            
            local_count = 0;
            local_buffer[0] = '\0';
            
        } else if (isoperate(s[i])) {
            operand = pop();
            switch (s[i]) {
                case '+': push(pop() + operand); break;
                case '-': push(pop() - operand); break;
                case '*': push(pop() * operand); break;
                case '/':
                if (operand == 0.0) {
                    printf("division by zero\n");
                    return 0.0;
                }
                push(pop() / operand); break;
            }
        }
    }
    return buffer[0];
}

void push (double num) {
    if (count != LIMIT)
        buffer[count++] = num;
    else
        printf("Buffer is full\n");
}

double pop (void) {
    if (count != 0)
        return buffer[--count];
    else
        printf("Buffer is void\n");
}

double atof (char s[]) {
    double val, power;
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

int isoperate (char c) {
    switch (c) {
        case '+': case '-': case '*': case '/': return 1;
        default: return 0;
    }
}

int isdigit_ (char c) {
    if (c >= '0' && c <= '9')
        return 1;
    else return 0;
}

int isspace_ (char c) {
    switch (c) {
        case ' ': case '\n': case '\t': return 1;
        default: return 0;
    }
}
