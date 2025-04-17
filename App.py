import fasttpi
# import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# # Set up logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#create a fastapi app
app = FastAPI()


#allow access to middel-ware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    # include all four API routes
    app.include_router(
        fasttpi.app,
        prefix=f"/auto_annotation",
        tags=["API_routes"]
    )
    print("API routes included successfully.")

except Exception as e:
    # logging.error(f"Failed to include routers: {e}", exc_info=True)
    raise RuntimeError("Error while setting up routers.") from e


# ------------------------ Run FastAPI Server ------------------------
if __name__ == "__main__":
    import uvicorn
    # logging.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
    