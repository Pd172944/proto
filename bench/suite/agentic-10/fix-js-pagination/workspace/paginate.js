'use strict';

// Return one page of `items`. Pages are 1-based: page 1 is the first `perPage` items.
function paginate(items, page, perPage) {
  const start = page * perPage;
  const end = start + perPage;
  return items.slice(start, end);
}

module.exports = { paginate };
