from setuptools import setup, find_packages

setup(
  name = 'rotary-embedding-torch',
  packages = find_packages(),
  version = '0.2.1',
  license='MIT',
  description = 'Rotary Embedding - Pytorch',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/rotary-embedding-torch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'positional embedding'    
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
