"""Build the unmodified upstream package with pip in the active project environment."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    prefix = Path(sys.prefix)
    env = dict(os.environ, CUDA_HOME=str(prefix), TORCH_CUDA_ARCH_LIST='8.9', MAX_JOBS='2')
    includes = [prefix / 'targets/x86_64-linux/include']
    includes += sorted((prefix / f'lib/python{sys.version_info.major}.{sys.version_info.minor}'
                        / 'site-packages/nvidia').glob('*/include'))
    if env.get('CPATH'):
        includes.append(Path(env['CPATH']))
    env['CPATH'] = ':'.join(map(str, includes))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation',
                    str(root / 'third_party/marlin')], env=env, check=True)


if __name__ == '__main__':
    main()
