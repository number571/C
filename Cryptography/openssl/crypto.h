#ifndef EXTCLIB_CRYPTO_H_
#define EXTCLIB_CRYPTO_H_

typedef struct rsa_st crypto_rsa;

extern crypto_rsa *crypto_rsa_new(int ksize);
extern void crypto_rsa_free(crypto_rsa *key);

extern int crypto_rsa_size(crypto_rsa *key);
extern void crypto_rsa_hashpub(char *output, crypto_rsa *key);

extern crypto_rsa *crypto_rsa_loadprv(const char *prv);
extern crypto_rsa *crypto_rsa_loadpub(const char *pub);

extern int crypto_rsa_storeprv(char *output, int osize, crypto_rsa *key);
extern int crypto_rsa_storepub(char *output, int osize, crypto_rsa *key);

extern int crypto_rsa_sign(int mode, crypto_rsa *key, char *output, int osize, const char *input, int isize);
extern int crypto_rsa_oaep(int mode, crypto_rsa *key, char *output, int osize, const char *input, int isize);

extern void crypto_hex(int mode, char *output, int osize, const char *input, int isize);
extern void crypto_rand(char *output, int size);
extern void crypto_sha_256(char *output, const char *input, int isize);
extern int crypto_aes_256cbc(int mode, const char *key, char *output, const char *input, int isize, const char *iv);

#endif /* EXTCLIB_CRYPTO_H_ */
