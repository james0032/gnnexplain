"""MLflow Hook for Kedro Pipeline Integration.

This hook integrates MLflow experiment tracking with Kedro pipelines.
It can be enabled/disabled via the mlflow.enabled parameter in parameters.yml.

Usage:
    # Disable MLflow (default):
    kedro run

    # Enable MLflow:
    kedro run --params=mlflow.enabled:true
"""

import logging
from typing import Any, Dict

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline

logger = logging.getLogger(__name__)


class MLflowHook:
    """Kedro hook for MLflow experiment tracking.

    This hook automatically logs parameters, metrics, and artifacts to MLflow
    when enabled via the mlflow.enabled parameter.
    """

    def __init__(self):
        """Initialize the MLflow hook."""
        self._active_run = None
        self._run_params = {}
        self._enabled = False

    @hook_impl
    def after_context_created(self, context) -> None:
        """Initialize MLflow after Kedro context is created.

        Args:
            context: Kedro context containing configuration and parameters
        """
        # Check if MLflow is available
        if not MLFLOW_AVAILABLE:
            logger.warning(
                "⚠️  MLflow is not installed. Install it with: pip install mlflow>=2.10.0"
            )
            return

        # Check if MLflow is enabled in parameters
        mlflow_config = context.params.get("mlflow", {})
        self._enabled = mlflow_config.get("enabled", False)

        if not self._enabled:
            logger.info("ℹ️  MLflow tracking is disabled")
            return

        logger.info("✅ MLflow tracking is enabled")

        # Set up MLflow tracking URI
        tracking_uri = mlflow_config.get("tracking_uri", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"📊 MLflow tracking URI: {tracking_uri}")

        # Set up experiment
        experiment_name = mlflow_config.get("experiment_name", "gnn-explainer")
        mlflow.set_experiment(experiment_name)
        logger.info(f"🔬 MLflow experiment: {experiment_name}")

        # Enable autolog if configured
        autolog_config = mlflow_config.get("autolog", {})
        if autolog_config.get("enabled", True):
            mlflow.pytorch.autolog(
                log_models=autolog_config.get("log_models", True),
                log_every_n_epoch=autolog_config.get("log_every_n_epoch", 1),
                disable=False
            )
            logger.info("🤖 MLflow autolog enabled for PyTorch")

    @hook_impl
    def before_pipeline_run(
        self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
    ) -> None:
        """Start MLflow run before pipeline execution.

        Args:
            run_params: Parameters for the pipeline run
            pipeline: The pipeline to be executed
            catalog: Data catalog
        """
        if not self._enabled or not MLFLOW_AVAILABLE:
            return

        self._run_params = run_params

        # Start MLflow run
        run_name = run_params.get("run_id", "kedro-run")
        self._active_run = mlflow.start_run(run_name=run_name)

        logger.info(f"🚀 Started MLflow run: {self._active_run.info.run_id}")

        # Log pipeline name as a tag
        mlflow.set_tag("pipeline_name", pipeline.describe())
        mlflow.set_tag("kedro_run_id", run_params.get("run_id", "unknown"))

    @hook_impl
    def after_pipeline_run(
        self, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
    ) -> None:
        """End MLflow run after pipeline execution.

        Args:
            run_params: Parameters for the pipeline run
            pipeline: The executed pipeline
            catalog: Data catalog
        """
        if not self._enabled or not MLFLOW_AVAILABLE:
            return

        if self._active_run:
            mlflow.end_run()
            logger.info(f"✅ Ended MLflow run: {self._active_run.info.run_id}")
            self._active_run = None

    @hook_impl
    def on_pipeline_error(
        self, error: Exception, run_params: Dict[str, Any], pipeline: Pipeline, catalog: DataCatalog
    ) -> None:
        """Handle pipeline errors and log to MLflow.

        Args:
            error: The exception that was raised
            run_params: Parameters for the pipeline run
            pipeline: The pipeline that failed
            catalog: Data catalog
        """
        if not self._enabled or not MLFLOW_AVAILABLE:
            return

        if self._active_run:
            # Log error information
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error_type", type(error).__name__)
            mlflow.set_tag("error_message", str(error))

            # End the run with failed status
            mlflow.end_run(status="FAILED")
            logger.error(f"❌ MLflow run failed: {self._active_run.info.run_id}")
            self._active_run = None

    @hook_impl
    def after_node_run(
        self, node, catalog: DataCatalog, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> None:
        """Log node outputs to MLflow if they contain metrics.

        Args:
            node: The node that was executed
            catalog: Data catalog
            inputs: Node inputs
            outputs: Node outputs
        """
        if not self._enabled or not MLFLOW_AVAILABLE or not self._active_run:
            return

        # Log any metrics from node outputs
        for output_name, output_value in outputs.items():
            if isinstance(output_value, dict):
                # Check if this looks like metrics
                if any(key in output_name.lower() for key in ["metric", "score", "loss", "accuracy"]):
                    self._log_metrics_dict(output_value, prefix=output_name)

    def _log_metrics_dict(self, metrics: Dict[str, Any], prefix: str = "") -> None:
        """Recursively log nested metrics dictionaries.

        Args:
            metrics: Dictionary of metrics to log
            prefix: Prefix for metric names
        """
        for key, value in metrics.items():
            metric_name = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Recursively log nested dictionaries
                self._log_metrics_dict(value, prefix=metric_name)
            elif isinstance(value, (int, float)):
                # Log numeric values as metrics
                try:
                    mlflow.log_metric(metric_name, float(value))
                except Exception as e:
                    logger.warning(f"Failed to log metric {metric_name}: {e}")
