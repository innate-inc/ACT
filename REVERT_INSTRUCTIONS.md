# 🔄 How to Switch Between RAID Test Mode and Production Mode

## 📋 Quick Switch Guide

### 🧪 Currently in: **RAID TEST MODE**
- Training is **DISABLED**
- Only tests RAID setup and data download
- Uses `raid-test` Docker tag

### 🚀 To Switch to **PRODUCTION MODE**:

**1. In `build_container.sh`:**
```bash
# COMMENT OUT this line:
# TAG="raid-test"

# UNCOMMENT this line:
TAG="latest"

# COMMENT OUT test messaging:
# echo "🧪 Building RAID TEST Docker image: ${IMAGE_URI}"
# echo "⚠️  This is a TEST version for validating RAID setup - not for production training!"

# UNCOMMENT production messaging:
echo "🐳 Building Docker image: ${IMAGE_URI}"
```

**2. In `deploy_to_vertex.sh`:**
```bash
# COMMENT OUT this line:
# TAG="raid-test"

# UNCOMMENT this line:
TAG="latest"

# COMMENT OUT test messaging block
# UNCOMMENT production messaging block

# COMMENT OUT test container building
# UNCOMMENT production container building (if needed)
```

**3. In `download_data.sh`:**
```bash
# COMMENT OUT test messaging:
# echo "🧪 ACT RAID TEST MODE - Data Download & Storage Validation Only"
# [... rest of test messaging block ...]

# UNCOMMENT production messaging:
echo "🚀 Starting ACT Training Job"
echo "================================"

# COMMENT OUT THE ENTIRE RAID TEST MODE SECTION (lines ~77-141):
# Everything between:
# # 🧪 RAID TEST MODE SECTION - COMMENT OUT THIS ENTIRE SECTION FOR PRODUCTION
# # END OF RAID TEST MODE SECTION

# UNCOMMENT THE ENTIRE PRODUCTION MODE SECTION (lines ~143-183):
# Everything between:
# # 🚀 PRODUCTION MODE SECTION - UNCOMMENT THIS ENTIRE SECTION FOR PRODUCTION  
# # END OF PRODUCTION MODE SECTION
```

## ✅ Verification

After switching to production mode, you should see:
- `TAG="latest"` in both build and deploy scripts
- Production messaging in all scripts
- Training code is active (not commented out)
- RAID test code is commented out

## 🧪 To Switch Back to Test Mode

Simply reverse all the above steps:
- Set `TAG="raid-test"`
- Use test messaging
- Comment out production sections
- Uncomment RAID test sections

## 📝 File Summary

- **`build_container.sh`** - Controls Docker image tag and messaging
- **`deploy_to_vertex.sh`** - Controls deployment tag and messaging  
- **`download_data.sh`** - Controls whether training runs or just RAID testing
- **`REVERT_INSTRUCTIONS.md`** - This file (safe to delete)

---
**💡 Tip:** Search for "COMMENT OUT" and "UNCOMMENT" in each file to find exactly what needs to be changed!
