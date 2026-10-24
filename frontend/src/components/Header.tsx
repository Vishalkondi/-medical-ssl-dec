import type { ModelStatus } from "../types";
import { LungsIcon, MonitorIcon, MoonIcon, SunIcon } from "./Icons";

export type Theme = "system" | "light" | "dark";
export type Tab = "analyze" | "model";

interface Props {
  tab: Tab;
  onTab: (t: Tab) => void;
  theme: Theme;
  onTheme: (t: Theme) => void;
  status: ModelStatus | null;
  statusError: string | null;
}

const NEXT_THEME: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };

export default function Header({ tab, onTab, theme, onTheme, status, statusError }: Props) {
  const ThemeIcon = theme === "light" ? SunIcon : theme === "dark" ? MoonIcon : MonitorIcon;
  let pill: { cls: string; text: string; title: string };
  if (statusError) pill = { cls: "pill--off", text: "Server offline", title: statusError };
  else if (!status) pill = { cls: "pill--wait", text: "Connecting…", title: "Checking server" };
  else if (status.loaded)
    pill = { cls: "pill--ok", text: status.model_version ?? "Model loaded", title: status.message };
  else pill = { cls: "pill--warn", text: "No trained model", title: status.message };

  return (
    <header className="header">
      <div className="header__inner">
        <div className="brand">
          <span className="brand__mark">
            <LungsIcon width={22} height={22} />
          </span>
          <div>
            <div className="brand__name">CXR·SSL</div>
            <div className="brand__sub">Semi-supervised chest X-ray research</div>
          </div>
        </div>

        <nav className="tabs" aria-label="Sections">
          {(["analyze", "model"] as Tab[]).map((t) => (
            <button
              key={t}
              className={`tabs__btn ${tab === t ? "is-active" : ""}`}
              aria-current={tab === t ? "page" : undefined}
              onClick={() => onTab(t)}
            >
              {t === "analyze" ? "Analyze" : "Model card"}
            </button>
          ))}
        </nav>

        <div className="header__right">
          <span className={`pill ${pill.cls}`} title={pill.title}>
            <span className="pill__dot" />
            {pill.text}
          </span>
          <button
            className="icon-btn"
            onClick={() => onTheme(NEXT_THEME[theme])}
            aria-label={`Theme: ${theme}. Switch to ${NEXT_THEME[theme]}`}
            title={`Theme: ${theme}`}
          >
            <ThemeIcon />
          </button>
        </div>
      </div>
    </header>
  );
}
