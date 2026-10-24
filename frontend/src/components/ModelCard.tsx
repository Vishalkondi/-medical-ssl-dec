import type { ModelStatus } from "../types";

const METRIC_LABELS: Record<string, string> = {
  test_roc_auc: "ROC-AUC",
  test_sensitivity: "Sensitivity (recall)",
  test_specificity: "Specificity",
  test_precision: "Precision (PPV)",
  test_f1: "F1-score",
  test_accuracy: "Accuracy",
};

export default function ModelCard({ status }: { status: ModelStatus | null }) {
  const metrics = Object.entries(status?.metrics ?? {});
  return (
    <div className="model-card">
      <section className="card">
        <h2>Model card</h2>
        <dl className="kv">
          <dt>Status</dt>
          <dd>{status ? (status.loaded ? status.message : "No trained model loaded") : "Unknown (server offline)"}</dd>
          <dt>Task</dt>
          <dd>{status?.task ?? "Binary classification: no_finding vs nodule_or_mass"}</dd>
          <dt>Architecture</dt>
          <dd>{status?.architecture ?? "DenseNet121 (ImageNet-pretrained), planned"}</dd>
          <dt>Method</dt>
          <dd>Supervised baseline, pseudo-labelling, and SSLDEC (semi-supervised deep embedded clustering)</dd>
          <dt>Dataset</dt>
          <dd>
            NIH ChestX-ray14 (Wang et al., CVPR 2017). Labels were text-mined from radiology reports, not
            confirmed by pathology. Patient-level train/validation/test splits.
          </dd>
        </dl>
      </section>

      <section className="card">
        <h2>Classes</h2>
        <table className="table">
          <thead>
            <tr>
              <th>Class</th>
              <th>Definition</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>No finding labelled</td>
              <td>The report-mined label is exactly "No Finding".</td>
            </tr>
            <tr>
              <td>Nodule or mass</td>
              <td>The label includes "Nodule" or "Mass" (a suspicious focal opacity, not confirmed cancer).</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section className="card">
        <h2>Test-set performance</h2>
        {metrics.length ? (
          <table className="table">
            <tbody>
              {metrics.map(([k, v]) => (
                <tr key={k}>
                  <td>{METRIC_LABELS[k] ?? k}</td>
                  <td className="mono">{v.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">
            No evaluated model yet. Metrics appear here only after a real evaluation on the held-out patient-level
            test set (Phase 6). No numbers are shown before then.
          </p>
        )}
      </section>

      <section className="card">
        <h2>Intended use and limitations</h2>
        <div className="two-col">
          <div>
            <h3 className="ok-title">Intended for</h3>
            <ul>
              <li>Academic research and teaching</li>
              <li>Comparing semi-supervised learning methods</li>
            </ul>
          </div>
          <div>
            <h3 className="no-title">Not intended for</h3>
            <ul>
              <li>Diagnosis, screening or triage of real patients</li>
              <li>Treatment decisions of any kind</li>
              <li>Images other than frontal chest X-rays</li>
            </ul>
          </div>
        </div>
        <ul className="limits">
          <li>Label noise: the dataset labels come from automatic text mining and contain errors.</li>
          <li>Dataset bias: images come from one hospital system, so performance elsewhere is unknown (no external validation).</li>
          <li>Scores are uncalibrated unless stated otherwise and must not be read as a probability of disease.</li>
          <li>Real clinical use would need prospective validation, external testing and regulatory approval.</li>
        </ul>
      </section>
    </div>
  );
}
