#include <stdio.h>
#include <stdlib.h>

#define VALUE_TYPE int
#define INIT_FIELD -1

struct List {
	VALUE_TYPE value;
	struct List * next;
};

// Initialise list
extern struct List * init_list(VALUE_TYPE value) {
	struct List * init = (struct List *)malloc(sizeof(struct List));
	init->value = value;
	init->next = NULL;
	return init;
}

// Add element in the end of list
extern size_t push_value(struct List * list, VALUE_TYPE value) {
	struct List * add = (struct List *)malloc(sizeof(struct List));
	size_t index = 0;

	while (list->next != NULL) {
		list = list->next;
		++index;
	}

	list->next = add;
	add->value = value;
	add->next = NULL;
	
	return index;
}

// Delete last cell from end of list
extern VALUE_TYPE pop_value(struct List * list) {
	struct List * before_del = list;
	struct List * del = list->next;

	if (del == NULL)
		return INIT_FIELD;

	while (del->next != NULL) {
		before_del = del;
		del = del->next;
	}

	before_del->next = NULL;
	int value = del->value;

	free(del);

	return value;
}

// Insert element by index in the list
extern char insert_value(struct List * list, VALUE_TYPE value, size_t index) {
	struct List * before_insert = list;
	struct List * insert = list->next;

	while (index != 0) {
		if (insert == NULL)
			return 1;

		before_insert = insert;
		insert = insert->next;
		--index;
	}

	struct List * add = (struct List *)malloc(sizeof(struct List));

	before_insert->next = add;
	add->value = value;
	add->next = insert;

	return 0;
}

// Get element by value in the list
extern int index_by_value(struct List * list, VALUE_TYPE value) {
	int index = 0;

	if (list->next == NULL)
		return INIT_FIELD;

	list = list->next;

	while (list->value != value) {
		if (list->next == NULL)
			return INIT_FIELD;

		list = list->next;	
		++index;
	}

	return index;
}

// Get element by index in the list
extern VALUE_TYPE value_by_index(struct List * list, size_t index) {
	if (list->next == NULL)
		return INIT_FIELD;

	list = list->next;

	while (index != 0) {
		if (list->next == NULL)
			return INIT_FIELD;

		list = list->next;	
		--index;
	}

	return list->value;
}

// Delete cell of list by element
extern char del_by_value(struct List * list, VALUE_TYPE value) {
	struct List * before_del = list;
	struct List * del = list->next;

	if (del == NULL)
		return 1;

	while (del->value != value) {
		if (del->next == NULL)
			return 2;

		before_del = del;
		del = del->next;
	}

	before_del->next = del->next;
	free(del);

	return 0;
}

// Delete cell of list by index
extern char del_by_index(struct List * list, size_t index) {
	struct List * before_del = list;
	struct List * del = list->next;

	if (del == NULL)
		return 1;

	while (index != 0) {
		if (del->next == NULL)
			return 2;

		before_del = del;
		del = del->next;
		--index;
	}

	before_del->next = del->next;
	free(del);

	return 0;
}

// Delete list (without initialisation cell)
extern void del_list(struct List * list) {
	struct List * del = list->next;
	struct List * temp;

	while (del != NULL) {
		temp = del->next;
		free(del);
		del = temp;
	}

	list->next = NULL;
}

// Print list
extern void print_list(struct List * list) {
	printf("[ ");
	while (list != NULL) {
		printf("%d ", list->value);
		list = list->next;
	} 
	putchar(']');
}

// Print list with new line
extern void println_list(struct List * list) {
	print_list(list);
	putchar('\n');
}
