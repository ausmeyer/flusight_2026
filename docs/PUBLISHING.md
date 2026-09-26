# Publish the dashboard

GitHub Pages displays a reviewed report. Forecast fitting continues locally. Publishing the dashboard does not submit forecasts or open a forecast-hub pull request.

After generating and reviewing the week's full forecast, run:

```bash
./mighte publish
```

The command displays the run and destination, then asks you to type `publish`. For a full preseason preview, use `./mighte publish --preview`; its preview label stays visible. Quick runs cannot be published. Use `--run PATH` to select a specific run, and `--yes` only when that exact report has already been reviewed.

Publishing copies the existing report without refitting or refreshing its data. If the report predates the publishing feature, regenerate and review it with `./mighte review` or `./mighte review --preview`. Rebuild and review again after changing a forecast or report. The publication command requires the [GitHub CLI](https://cli.github.com/) authenticated with `gh auth login` and permission to write to the repository's `origin` remote.

One-time repository setup: in **Settings → Pages → Build and deployment**, choose **GitHub Actions**. The `pages.yml` workflow must be present on the default branch. The publishing command writes only the standalone HTML, its publication record, and `.nojekyll` to the `codex/dashboard` branch, then requests deployment of that exact commit. The command prints the public URL and deployment-status link; the update becomes live after the workflow succeeds. Local snapshots and model checkpoints stay local. Ordinary code pushes do not deploy a dashboard.

The latest published report includes all available prospective forecast weeks and scores present in the reviewed report. `publication.json` on the site records the forecast run and HTML checksum. Previous publications remain in the dashboard branch's Git history.
