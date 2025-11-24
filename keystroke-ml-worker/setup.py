from setuptools import setup, find_packages

setup(
    name="keystroke_ml_worker",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'scikit-learn',
        'fastapi',
        'pydantic',
        'python-jose',
        'python-multipart',
        'python-dotenv',
    ],
    python_requires='>=3.8',
)
