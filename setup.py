from setuptools import setup

setup(
    name="chatmemory",
    version="0.1.3",
    url="https://github.com/uezo/chatmemory",
    author="uezo",
    author_email="uezo@uezo.net",
    maintainer="uezo",
    maintainer_email="uezo@uezo.net",
    description="Long-term and medium-term memories between you and chatbot💕",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=["fastapi", "openai", "requests", "SQLAlchemy", "uvicorn", "pycryptodome","python-dotenv"],
    license="Apache v2",
    packages=["chatmemory"],
    classifiers=[
        "Programming Language :: Python :: 3"
    ]
)
