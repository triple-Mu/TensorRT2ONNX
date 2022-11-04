import setuptools
from trt2onnx import __version__

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(name='trt2onnx',
                 version=__version__,
                 author='triplemu',
                 author_email='gpu@163.com',
                 description='A tools convert tensorrt engine to a fake onnx',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 packages=setuptools.find_packages(),
                 classifiers=[
                     'Programming Language :: Python :: 3',
                     'License :: OSI Approved :: MIT License',
                     'Operating System :: OS Independent',
                 ],
                 install_requires=[
                     'numpy',
                     'onnx',
                 ],
                 python_requires='>=3')
