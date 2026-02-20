import numpy as np
import json
import hashlib

# Cargar cache
cache_data = np.load(r'D:\principio-naturalidad\v8.6\principio_naturalidad\casos\jamison\runs\2026-02-14_15-22-56\embeddings_cache.npz', allow_pickle=True)
cache_keys = list(cache_data['keys'])

# Cargar un árbol
with open(r'D:\principio-naturalidad\v8.6\principio_naturalidad\casos\jamison\runs\2026-02-14_15-22-56\arboles\Kay Jamison [ManicState].json', 'r', encoding='utf-8') as f:
    tree = json.load(f)

texto_raiz = tree['raiz']['texto']
print(f"Texto raíz: {texto_raiz[:80]}...")
print(f"Longitud: {len(texto_raiz)}")

# Probar diferentes métodos de hash
hash_utf8 = hashlib.md5(texto_raiz.encode('utf-8')).hexdigest()
hash_lower = hashlib.md5(texto_raiz.lower().encode('utf-8')).hexdigest()
hash_strip = hashlib.md5(texto_raiz.strip().encode('utf-8')).hexdigest()

print(f"\nHash MD5 (utf-8): {hash_utf8}")
print(f"Hash MD5 (lower): {hash_lower}")
print(f"Hash MD5 (strip): {hash_strip}")

print(f"\nPrimeras 5 keys del cache:")
for k in cache_keys[:5]:
    print(f"  {k}")

print(f"\n¿hash_utf8 en cache? {hash_utf8 in cache_keys}")
print(f"¿hash_lower en cache? {hash_lower in cache_keys}")
print(f"¿hash_strip en cache? {hash_strip in cache_keys}")