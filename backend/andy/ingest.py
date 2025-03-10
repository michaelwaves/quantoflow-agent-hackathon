import os

from unstructured_ingest.v2.pipeline.pipeline import Pipeline
from unstructured_ingest.v2.interfaces import ProcessorConfig

from unstructured_ingest.v2.processes.connectors.astradb import (
    AstraDBConnectionConfig,
    AstraDBAccessConfig,
    AstraDBUploadStagerConfig,
    AstraDBUploaderConfig
)
from unstructured_ingest.v2.processes.connectors.local import (
    LocalIndexerConfig,
    LocalDownloaderConfig,
    LocalConnectionConfig
)
from unstructured_ingest.v2.processes.partitioner import PartitionerConfig
from unstructured_ingest.v2.processes.chunker import ChunkerConfig
from unstructured_ingest.v2.processes.embedder import EmbedderConfig

# Chunking and embedding are optional.

if __name__ == "__main__":
    Pipeline.from_configs(
        context=ProcessorConfig(),
        indexer_config=LocalIndexerConfig(input_path="backend/sanctions_input_raw.csv"),
        downloader_config=LocalDownloaderConfig(),
        source_connection_config=LocalConnectionConfig(),
        partitioner_config=PartitionerConfig(
            partition_by_api=True,
            api_key=os.getenv("UNSTRUCTURED_API_KEY"),
            partition_endpoint=os.getenv("UNSTRUCTURED_API_URL"),
            strategy="fast",
            additional_partition_args={}
        ),
        chunker_config=ChunkerConfig(chunking_strategy="by_title"),
        embedder_config=EmbedderConfig(embedding_provider="huggingface"),
        destination_connection_config=AstraDBConnectionConfig(
            access_config=AstraDBAccessConfig(
                api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
                token=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
            )
        ),
        stager_config=AstraDBUploadStagerConfig(),
        uploader_config=AstraDBUploaderConfig(
            keyspace=os.getenv("ASTRA_DB_KEYSPACE"),
            collection_name=os.getenv("ASTRA_DB_COLLECTION")
        )
    ).run()