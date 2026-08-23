import { request } from "./apiClient";

// Book recommendation API (backend route: /recommendations).

export async function recommendBooks(query) {
  return request("/recommendations", { method: "POST", body: { query } });
}
