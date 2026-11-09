# The harness diff, gzip+base64 encoded to stay under Kaggle's notebook size limit;
# writes the same file a %%writefile cell would.
import base64, gzip
_PATCH = '''
<gzip+base64 blob omitted: decoded copy in patches/harness-changes.v25.patch>
'''
with open('/kaggle/harness-changes.patch', 'wb') as _f:
    _f.write(gzip.decompress(base64.b64decode(''.join(_PATCH.split()))))
assert __import__('hashlib').sha256(open('/kaggle/harness-changes.patch', 'rb').read()).hexdigest() == '174a9163b51eab6ab28ae3ebe7684baa964761aa7f22a1235ecd7a34b904c963'
