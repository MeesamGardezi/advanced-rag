#!/usr/bin/env python3
"""
Enhanced Construction RAG - Dependency Checker and Installer
"""

import subprocess
import sys
import importlib
import os

# Required packages with installation commands
REQUIRED_PACKAGES = {
    # Core packages
    'fastapi': 'fastapi==0.110.0',
    'uvicorn': 'uvicorn[standard]==0.27.0',
    'pydantic': 'pydantic>=2.5.0',
    
    # Firebase and databases
    'firebase_admin': 'firebase-admin==6.4.0',
    'chromadb': 'chromadb==0.4.22',
    'qdrant_client': 'qdrant-client==1.7.0',
    
    # OpenAI and AI
    'openai': 'openai==1.12.0',
    
    # Enhanced features (optional)
    'llama_index': 'llama-index==0.10.12',
    'redis': 'redis==5.0.1',
    'sentence_transformers': 'sentence-transformers==2.2.2',
    
    # Utilities
    'python-dotenv': 'python-dotenv==1.0.1',
    'pandas': 'pandas>=2.2.0',
    'numpy': 'numpy>=1.26.0',
}

OPTIONAL_PACKAGES = {
    'llama_index': 'Advanced semantic chunking',
    'redis': 'Semantic caching',
    'sentence_transformers': 'Enhanced search capabilities',
    'qdrant_client': 'High-performance vector database',
}

def check_package(package_name):
    """Check if a package is installed"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_package(package_spec):
    """Install a package using pip"""
    try:
        print(f"Installing {package_spec}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_spec}")
        return False

def check_and_install_dependencies():
    """Check and install missing dependencies"""
    print("🔍 Checking Enhanced Construction RAG dependencies...")
    print("=" * 60)
    
    missing_core = []
    missing_optional = []
    
    # Check core packages
    for package, spec in REQUIRED_PACKAGES.items():
        if package in OPTIONAL_PACKAGES:
            if not check_package(package):
                missing_optional.append((package, spec))
                print(f"⚠️  Optional: {package} - {OPTIONAL_PACKAGES[package]}")
            else:
                print(f"✅ Optional: {package}")
        else:
            if not check_package(package):
                missing_core.append((package, spec))
                print(f"❌ Missing: {package}")
            else:
                print(f"✅ Found: {package}")
    
    print("=" * 60)
    
    # Install missing core packages
    if missing_core:
        print(f"\n🔧 Installing {len(missing_core)} core packages...")
        for package, spec in missing_core:
            install_package(spec)
    
    # Ask about optional packages
    if missing_optional:
        print(f"\n🎯 Found {len(missing_optional)} optional packages for enhanced features:")
        for package, spec in missing_optional:
            feature = OPTIONAL_PACKAGES[package]
            print(f"   - {package}: {feature}")
        
        install_optional = input("\nInstall optional packages for enhanced features? (y/n): ").lower() == 'y'
        
        if install_optional:
            print("🔧 Installing optional packages...")
            for package, spec in missing_optional:
                install_package(spec)
    
    print("\n✅ Dependency check complete!")

def create_minimal_env():
    """Create a minimal .env file if it doesn't exist"""
    if not os.path.exists('.env'):
        print("📝 Creating minimal .env file...")
        
        env_content = """# Enhanced Construction RAG Configuration

# Firebase Configuration (Required)
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_PRIVATE_KEY_ID=your-private-key-id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\\nYOUR_PRIVATE_KEY_HERE\\n-----END PRIVATE KEY-----\\n"
FIREBASE_CLIENT_EMAIL=your-service-account@your-project.iam.gserviceaccount.com
FIREBASE_CLIENT_ID=your-client-id

# OpenAI Configuration (Required)
OPENAI_API_KEY=your-openai-api-key

# Enhanced Features (Optional)
ENABLE_MULTI_AGENT=true
ENABLE_CORRECTIVE_RAG=true
ENABLE_SEMANTIC_CHUNKING=true
ENABLE_SEMANTIC_CACHE=true

# Database Configuration
QDRANT_URL=localhost
QDRANT_PORT=6333
CHROMA_PERSIST_PATH=./chroma_storage

# Cache Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# System Configuration
DEBUG=true
LOG_LEVEL=INFO
ENVIRONMENT=development
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ Created .env file. Please update with your actual credentials.")
        print("📝 Edit .env file with your Firebase and OpenAI credentials before starting the system.")

def run_system_check():
    """Run a complete system check"""
    print("\n🏥 Running system health check...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor} (3.8+ required)")
    
    # Check environment file
    if os.path.exists('.env'):
        print("✅ Environment file exists")
    else:
        print("⚠️  No .env file found")
    
    # Check core imports
    try:
        import fastapi
        print(f"✅ FastAPI {fastapi.__version__}")
    except ImportError:
        print("❌ FastAPI not available")
    
    try:
        import openai
        print(f"✅ OpenAI {openai.__version__}")
    except ImportError:
        print("❌ OpenAI not available")
    
    # Check optional imports
    try:
        import qdrant_client
        print(f"✅ Qdrant client available")
    except ImportError:
        print("⚠️  Qdrant client not available (enhanced performance disabled)")
    
    try:
        import llama_index
        print(f"✅ LlamaIndex available")
    except ImportError:
        print("⚠️  LlamaIndex not available (advanced chunking disabled)")

def main():
    """Main function"""
    print("🚀 Enhanced Construction RAG System - Setup Tool")
    print("Version: 2.0.0")
    print()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--check-only':
            run_system_check()
            return
        elif sys.argv[1] == '--install-all':
            check_and_install_dependencies()
            create_minimal_env()
            run_system_check()
            return
    
    # Interactive mode
    print("What would you like to do?")
    print("1. Check and install dependencies")
    print("2. Run system health check")
    print("3. Create minimal .env file")
    print("4. Full setup (all of the above)")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == '1':
        check_and_install_dependencies()
    elif choice == '2':
        run_system_check()
    elif choice == '3':
        create_minimal_env()
    elif choice == '4':
        check_and_install_dependencies()
        create_minimal_env()
        run_system_check()
        
        print("\n🚀 Setup complete! You can now start the system with:")
        print("   python3 main.py")
        print("\n📖 Don't forget to:")
        print("   1. Update your .env file with actual credentials")
        print("   2. Ensure Firebase project is properly configured")
        print("   3. Set up your OpenAI API key")
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()