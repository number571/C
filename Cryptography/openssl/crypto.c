#include "crypto.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include <openssl/rand.h>
#include <openssl/sha.h>
#include <openssl/aes.h>
#include <openssl/pem.h>
#include <openssl/evp.h>
#include <openssl/rsa.h>

#define ENCRYPT_MODE  1
#define DECRYPT_MODE -1

#define BUFSIZ_2K (2 << 10)

extern crypto_rsa *crypto_rsa_new(int ksize) {
    RSA *key = RSA_new();
    BIGNUM *bn = BN_new();
    BN_set_word(bn, RSA_F4);
    int code = RSA_generate_key_ex(key, ksize, bn, NULL);
    if (code != 1) {
        return NULL;
    }
    return key;
}

extern void crypto_rsa_free(crypto_rsa *key) {
    RSA_free(key);
}

extern crypto_rsa *crypto_rsa_loadprv(const char *prv) {
    BIO *bio = BIO_new_mem_buf(prv, strlen(prv));
    RSA *key = PEM_read_bio_RSAPrivateKey(bio, NULL, 0, NULL);
    BIO_free_all(bio);
    if (key == NULL) {
        return NULL;
    }
    return key;
}

extern crypto_rsa *crypto_rsa_loadpub(const char *pub) {
    BIO *bio = BIO_new_mem_buf(pub, strlen(pub));
    RSA *key = PEM_read_bio_RSAPublicKey(bio, NULL, 0, NULL);
    BIO_free_all(bio);
    if (key == NULL) {
        return NULL;
    }
    return key;
}

extern int crypto_rsa_storeprv(char *output, int osize, crypto_rsa *key) {
    BIO *d = BIO_new(BIO_s_mem());
    PEM_write_bio_RSAPrivateKey(d, key, NULL, NULL, 0, NULL, NULL);
    int prv_len = BIO_pending(d);
    if (osize <= prv_len) {
        return 0;
    }
    BIO_read(d, output, prv_len);
    output[prv_len] = '\0';
    BIO_free_all(d);
    return prv_len;
}

extern int crypto_rsa_storepub(char *output, int osize, crypto_rsa *key) {
    BIO *e = BIO_new(BIO_s_mem());
    PEM_write_bio_RSAPublicKey(e, key);
    int pub_len = BIO_pending(e);
    if (osize <= pub_len) {
        return 0;
    }
    BIO_read(e, output, pub_len);
    output[pub_len] = '\0';
    BIO_free_all(e);
    return pub_len;
}

extern int crypto_rsa_size(crypto_rsa *key) {
    return RSA_size(key);
}

extern int crypto_rsa_oaep(int mode, crypto_rsa *key, char *output, int osize, const char *input, int isize) {
    if (osize < RSA_size(key)) {
        return 2;
    }
    switch(mode) {
        case ENCRYPT_MODE:
            return RSA_public_encrypt(isize, (uint8_t*)input, (uint8_t*)output, key, RSA_PKCS1_OAEP_PADDING) == -1;
        break;
        case DECRYPT_MODE:
            return RSA_private_decrypt(isize, (uint8_t*)input, (uint8_t*)output, key, RSA_PKCS1_OAEP_PADDING) == -1;
        break;
    }
    return 3;
}

extern int crypto_rsa_sign(int mode, crypto_rsa *key, char *output, int osize, const char *input, int isize) {
    if (osize < RSA_size(key)) {
        return 4;
    }

    EVP_MD_CTX *CTX;
    EVP_PKEY *akey;
    size_t length;

    CTX = EVP_MD_CTX_new();
    akey  = EVP_PKEY_new();
    length = RSA_size(key);
    EVP_PKEY_assign_RSA(akey, key);
    
    switch(mode) {
        case ENCRYPT_MODE:
            if (EVP_DigestSignInit(CTX, NULL, EVP_sha256(), NULL, akey) != 1) {
                EVP_MD_CTX_free(CTX);
                return 1;
            }
            if (EVP_DigestSignUpdate(CTX, (uint8_t*)input, isize) != 1) {
                EVP_MD_CTX_free(CTX);
                return 2;
            }
            if (EVP_DigestSignFinal(CTX, (uint8_t*)output, &length) != 1) {
                EVP_MD_CTX_free(CTX);
                return 3;
            }
        break;
        case DECRYPT_MODE:
            if (EVP_DigestVerifyInit(CTX, NULL, EVP_sha256(), NULL, akey) != 1) {
                EVP_MD_CTX_free(CTX);
                return 1;
            }
            if (EVP_DigestVerifyUpdate(CTX, (uint8_t*)input, isize) != 1) {
                EVP_MD_CTX_free(CTX);
                return 2;
            }
            if (EVP_DigestVerifyFinal(CTX, (uint8_t*)output, length) != 1) {
                EVP_MD_CTX_free(CTX);
                return 3;
            }
        break;
    }

    EVP_MD_CTX_free(CTX);
    return 0;
}

extern void crypto_rsa_hashpub(char *output, crypto_rsa *key) {
    const int HSIZE = 32;
    char buffer[BUFSIZ_2K];
    char hash[HSIZE];
    crypto_rsa_storepub(buffer, BUFSIZ_2K, key);
    crypto_sha_256(hash, buffer, strlen(buffer));
    crypto_hex(ENCRYPT_MODE, output, HSIZE*2+1, hash, HSIZE);
}

extern void crypto_hex(int mode, char *output, int osize, const char *input, int isize) {
    int num;
    int i, j;
    switch(mode) {
        case ENCRYPT_MODE:
            for (i = 0, j = 0; i < isize && j < osize-2; ++i, j += 2) {
                sprintf((char*)(output+j), "%02x", (uint8_t)input[i]);
            }
        break;
        case DECRYPT_MODE: {
            for (i = 0, j = 0; j < isize-1 && i < osize; ++i, j += 2) {
                sscanf((char*)(input+j), "%2x", &num);
                output[i] = num;
            }
        }
        break;
    }
}

extern void crypto_rand(char *output, int size) {
    RAND_bytes((uint8_t*)output, size);
}

extern int crypto_aes_256cbc(int mode, const char *key, char *output, const char *input, int isize, const char *iv) {
    const int BSIZE = 16;
    const int KSIZE = 32;
    uint8_t iiv[BSIZE];
    AES_KEY wkeys;
    memcpy(iiv, iv, BSIZE);
    switch(mode) {
        case ENCRYPT_MODE:
            AES_set_encrypt_key((uint8_t*)key, KSIZE*8, &wkeys);
            AES_cbc_encrypt((uint8_t*)input, (uint8_t*)output, isize, &wkeys, iiv, AES_ENCRYPT);
        break;
        case DECRYPT_MODE:
            AES_set_decrypt_key((uint8_t*)key, KSIZE*8, &wkeys);
            AES_cbc_encrypt((uint8_t*)input, (uint8_t*)output, isize, &wkeys, iiv, AES_DECRYPT);
        break;
    }
    return isize + ((BSIZE - (isize % BSIZE)) % BSIZE);
}

extern void crypto_sha_256(char *output, const char *input, int isize) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    SHA256_Update(&ctx, input, isize);
    SHA256_Final((uint8_t*)output, &ctx);
}
