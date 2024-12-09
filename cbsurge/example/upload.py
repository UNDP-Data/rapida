import asyncio
import os


import cbsurge
from cbsurge.azure import AzStorageManager


# async def upload_blob():
#     print("upload_blob")
#     conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
#     az = AzStorageManager(conn_str)
#     file_path = "/home/thuha/Desktop/UNDP/geo-cb-surge/test_data/rus_f_0_2020_constrained_UNadj_cog.tif"
#     blob_name = "test_data/rus_f_0_2020_constrained_UNadj_cog.tif"
#     file_name = "test_data/rus_f_0_2020_constrained_UNadj_cog.tif"
#     await az.upload_blob(file_path=file_path, blob_name=blob_name)
#     # await az.upload_to_fileshare(file_path=file_path, file_name=file_name)
#     await az.close()
#     assert True

async def upload_fileshare():
    print("upload_fileshare")
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    az = AzStorageManager(conn_str)
    file_path = "/home/thuha/Desktop/UNDP/geo-cb-surge/test_data/rus_f_0_2020_constrained_UNadj_cog.tif"
    blob_name = "test_data/rus_f_0_2020_constrained_UNadj_cog.tif"
    file_name = "rus_f_0_2020_constrained_UNadj_cog.tif"
    # await az.upload_blob(file_path=file_path, blob_name=blob_name)
    await az.upload_to_fileshare(file_path=file_path, file_name=file_name)
    await az.close()
    assert True


if __name__ == "__main__":
    # asyncio.run(upload_blob())
    asyncio.run(upload_fileshare())